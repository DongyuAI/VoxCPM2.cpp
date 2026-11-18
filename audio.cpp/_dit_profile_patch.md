## VoxCPM2 DiT estimator 埋点补丁说明

目标：定位每次 estimator forward（约 9ms）的耗时构成：
  - upload：5 次 tensor_set 的 H2D
  - compute：compute_backend_graph
  - sync：ggml_backend_synchronize
  - readback：tensor_get 的 D2H

实现方式：在 VoxCPM2DiTEstimatorRuntime::Impl 里加静态累加器（字节级精确插入），
在 VoxCPM2CFMRuntime::Impl::generate_patch 返回前输出一次汇总（每次 generate_patch
对应一个 patch 的 6 步扩散，共约 6 次 estimator run）。用环境变量 VOXCPM2_DIT_PROFILE=1 控制。

注意：这是诊断用补丁，默认关闭时对数值零影响（只多了几个 if 分支和时间点读取）。
诊断完成后可 revert 或保留（关闭状态）。

=== 具体改动（行号基于 baseline-upstream tag）===

[1] estimator run() 中，把 tensor_set 段包上计时：
    在 "ggml_backend_tensor_set(x_, ..." 前插入：
        const auto _t_upload0 = Clock::now();
    在 5 次 tensor_set 之后、"engine::core::set_backend_threads" 前插入：
        g_dit_prof.upload_ms += engine::debug::elapsed_ms(_t_upload0, Clock::now());

[2] compute + sync 分段：
    把
        const ggml_status status = engine::core::compute_backend_graph(...);
        ggml_backend_synchronize(...);
    改为
        const auto _t_compute0 = Clock::now();
        const ggml_status status = engine::core::compute_backend_graph(...);
        const auto _t_sync0 = Clock::now();
        g_dit_prof.compute_ms += engine::debug::elapsed_ms(_t_compute0, _t_sync0);
        ggml_backend_synchronize(...);
        g_dit_prof.sync_ms += engine::debug::elapsed_ms(_t_sync0, Clock::now());

[3] readback 计时：
    在 "std::vector<float> output..." 前插入 const auto _t_read0 = Clock::now();
    在 tensor_get 后插入 g_dit_prof.readback_ms += ...; g_dit_prof.calls++;

[4] 文件顶部匿名 namespace 加静态结构：
    struct DiTProfile { double upload_ms=0, compute_ms=0, sync_ms=0, readback_ms=0; long calls=0; };
    static DiTProfile g_dit_prof;

[5] generate_patch 的 "return channel_major_to_patch_major(x);" 前插入：
    if (g_dit_prof.calls && std::getenv("VOXCPM2_DIT_PROFILE")) {
        debug::timing_log_scalar("voxcpm2.dit.upload_ms", g_dit_prof.upload_ms);
        debug::timing_log_scalar("voxcpm2.dit.compute_ms", g_dit_prof.compute_ms);
        debug::timing_log_scalar("voxcpm2.dit.sync_ms", g_dit_prof.sync_ms);
        debug::timing_log_scalar("voxcpm2.dit.readback_ms", g_dit_prof.readback_ms);
        debug::timing_log_scalar("voxcpm2.dit.calls", g_dit_prof.calls);
        g_dit_prof = DiTProfile{};
    }
