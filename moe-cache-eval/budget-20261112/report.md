# Shared-cache P0 budget evaluation

Memory bandwidth assumption: 153 GB/s (Apple M5 specification).

| Requested MiB | Actual MiB | Decode mean +/- sd tok/s | Before-load hit % | Sync bundles/token | Read MiB/token | Pread ms/token | Total ms/token | Footprint MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 256 | 5.31 +/- 0.53 | 3.39 | 309.15 | 951.42 | 100.07 | 189.72 | 530.8 |
| 512 | 512 | 6.08 +/- 0.73 | 23.63 | 244.37 | 653.02 | 80.75 | 166.16 | 786.4 |
| 1024 | 1024 | 7.81 +/- 0.22 | 41.39 | 187.54 | 393.33 | 51.11 | 128.05 | 1297.9 |
| 2048 | 2048 | 7.33 +/- 0.58 | 56.40 | 139.52 | 255.12 | 53.58 | 136.91 | 2322.5 |
| 4096 | 4096 | 7.66 +/- 0.35 | 75.80 | 77.43 | 141.34 | 50.25 | 130.70 | 4369.1 |
| 6144 | 6144 | 8.15 +/- 0.25 | 84.73 | 48.87 | 89.37 | 39.84 | 122.80 | 6415.5 |

Each point averages repeated runs; all runs use one fixed prompt. Process disk bytes are OS-attributed process storage reads, not expert-specific SSD misses.

DRAM-only floor = (estimated active weight bytes + 2 * logical file-read bytes/token) / specified bandwidth. This optimistic traffic model excludes quantization, graph synchronization, KV and scheduling costs.

Raw data: records.json; model/hardware: manifest.json; per-run logs and time output are retained.

Application-level I/O calibration: 2.33 GB/s for 14 storage-dominated samples; 16.08 GB/s for 242 zero-disk samples. These are not device peak specifications.

Full-model mapping reference: [30.92, 31.6, 32.11] tok/s; mean 31.54 tok/s.
