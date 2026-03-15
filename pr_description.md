## Description

This PR introduces parallel Basin-Hopping sampling using `joblib`, allowing for speedups during LON construction by running multiple independent sampling runs concurrently across multiple CPU cores. The PR also includes some API change and default parameter tuning.

I tried to keep the structure of the `sampling.py` as close to previous implementation as possible, but running the sampling procedure in separate threads required some greater changes. Most notibly, creating the `_run_single_bh_in_worker` function, which is needed because of how joblib handles pararellism - the function run by the multiprocessing runner has to be memory-independent from other objects in the file, so creating another function instead of method and creating a new `BasinHoppingSampler` object was required.

The pararell sampling procedure was benchmarked against the single-threaded and other implementation, which was discussed internally. It proves to be efficient enough.

```
Sequential vs Old-Processes-Approach vs New-Processes-Approach Basin-Hopping
(mean of 3 repetitions, n_jobs=-1, processes backend)

Scenario                      Seq (s) OldProc (s) NewProc (s) OldProc/Seq NewProc/Seq    New/Old
------------------------------------------------------------------------------------------------
sphere      2-D  50-runs       13.031       1.687       0.805        7.73x       16.18x       2.09x
rastrigin   2-D  50-runs       14.159       0.766       0.768       18.50x       18.43x       1.00x
rosenbrock  2-D  50-runs       38.180       2.172       2.202       17.57x       17.34x       0.99x
ackley      2-D  50-runs       41.846       2.402       2.102       17.42x       19.91x       1.14x
sphere      5-D  50-runs       14.829       0.934       0.891       15.87x       16.65x       1.05x
sphere     10-D  50-runs       21.189       1.275       1.226       16.62x       17.28x       1.04x
rastrigin  10-D  50-runs       52.352       2.538       2.339       20.63x       22.39x       1.09x
rosenbrock 10-D  50-runs      100.985       5.954       5.383       16.96x       18.76x       1.11x
ackley     10-D  50-runs       74.931       3.569       3.678       21.00x       20.37x       0.97x
sphere     10-D  10-runs        4.277       0.806       0.794        5.30x        5.38x       1.02x
sphere     10-D 200-runs       73.619       2.317       2.192       31.77x       33.58x       1.06x
rastrigin  10-D 200-runs      211.751       5.401       4.760       39.21x       44.49x       1.13x
ackley     10-D 200-runs      321.181       7.331       6.818       43.81x       47.11x       1.08x
sphere      2-D  10-runs        2.692       0.387       0.424        6.95x        6.36x       0.91x
sphere      2-D 200-runs       51.139       1.600       1.642       31.97x       31.15x       0.97x
------------------------------------------------------------------------------------------------

n_jobs scaling — sphere 2-D, 50 runs (processes backend)
(mean of 3 repetitions)

n_jobs     OldProc (s) NewProc (s)    Old/Seq    New/Seq    New/Old
-------------------------------------------------------------------
sequential (reference)                  1.00x      1.00x        N/A
1               12.358      12.397       1.05x       1.05x       1.00x
2                7.053       6.519       1.84x       1.99x       1.08x
4                3.764       3.398       3.45x       3.82x       1.11x
8                2.415       2.087       5.38x       6.22x       1.16x
-1 (all)         1.780       0.806       7.30x      16.11x       2.21x
-------------------------------------------------------------------
```

```
BENCHMARK CPU SPEC

Architecture:             x86_64
  CPU op-mode(s):         32-bit, 64-bit
  Address sizes:          48 bits physical, 48 bits virtual
  Byte Order:             Little Endian
CPU(s):                   128
  On-line CPU(s) list:    0-127
Vendor ID:                AuthenticAMD
  Model name:             AMD EPYC 7763 64-Core Processor
    CPU family:           25
    Model:                1
    Thread(s) per core:   2
    Core(s) per socket:   64
    Socket(s):            1
    Stepping:             1
    Frequency boost:      enabled
    CPU(s) scaling MHz:   46%
    CPU max MHz:          3529.0520
    CPU min MHz:          1500.0000
    BogoMIPS:             4900.15
    Flags:                fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse s
                          se2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl nonstop_tsc cpu
                          id extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 movbe p
                          opcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignss
                          e 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwai
                          tx cpb cat_l3 cdp_l3 invpcid_single hw_pstate ssbd mba ibrs ibpb stibp vmmcall fsgsbase bmi1 a
                          vx2 smep bmi2 erms invpcid cqm rdt_a rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xg
                          etbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local clzero irperf xsaveerptr rdpru 
                          wbnoinvd amd_ppin brs arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decode
                          assists pausefilter pfthreshold v_vmsave_vmload vgif v_spec_ctrl umip pku ospke vaes vpclmulqd
                          q rdpid overflow_recov succor smca fsrm
Virtualization features:  
  Virtualization:         AMD-V
Caches (sum of all):      
  L1d:                    2 MiB (64 instances)
  L1i:                    2 MiB (64 instances)
  L2:                     32 MiB (64 instances)
  L3:                     256 MiB (8 instances)
NUMA:                     
  NUMA node(s):           1
  NUMA node0 CPU(s):      0-127
Vulnerabilities:          
  Gather data sampling:   Not affected
  Itlb multihit:          Not affected
  L1tf:                   Not affected
  Mds:                    Not affected
  Meltdown:               Not affected
  Mmio stale data:        Not affected
  Reg file data sampling: Not affected
  Retbleed:               Not affected
  Spec rstack overflow:   Mitigation; safe RET, no microcode
  Spec store bypass:      Mitigation; Speculative Store Bypass disabled via prctl
  Spectre v1:             Mitigation; usercopy/swapgs barriers and __user pointer sanitization
  Spectre v2:             Mitigation; Retpolines; IBPB conditional; IBRS_FW; STIBP always-on; RSB filling; PBRSB-eIBRS N
                          ot affected; BHI Not affected
  Srbds:                  Not affected
  Tsx async abort:        Not affected
```


## Changes Made

### `src/lonkit/sampling.py`

#### `BasinHoppingSamplerConfig`
- Added `n_jobs: int | None = 1` parameter to control parallel execution.

#### `BasinHoppingSampler`
- Refactored into modular methods:
  - `_single_bh_run()`: Executes a single Basin-Hopping run
  - `_sequential_bh()`: Runs sequentially (when `effective_jobs=1`)
  - `_parallel_bh()`: Runs in parallel using `joblib` (when `effective_jobs != 1`)
- Added `verbose: bool = False` parameter to `sample()` for progress display using `tqdm`
- `_perturbation()` now accepts `rng` as parameter instead of using instance-level RNG
- the instance level `_rng` was dropped, because of the need to have reproducible approach across the processes, which required `SeedSequence`

#### Module-level worker
- Added `_run_single_bh_in_worker()` module-level function for `joblib` pickling compatibility

### `src/lonkit/step_size.py`
- Updated `_perturbation()` calls to pass `rng` parameter for consistency with sampling changes

### `tests/test_parallel_sampling.py` (new file)
- Added comprehensive tests for parallel reproducibility:

### Documentation

- docs changes

### Dependencies
- Added `joblib>=1.3.0` as a required dependency
