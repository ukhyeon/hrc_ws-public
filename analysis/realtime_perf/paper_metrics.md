# Table 1. ROS 2 node operation runtime

| Node | Operation | Mean [ms] | Std. [ms] | P95 [ms] | Max [ms] | Equivalent Processing Rate [Hz] | N |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Human sensing | Human pose estimation | 33.35 | 2.09 | 36.72 | 62.31 | 29.98 | 1526 |
| Human dynamics | Human dynamics state update | 32.86 | 4.06 | 38.95 | 55.58 | 30.43 | 1527 |
| Robot dynamics | Robot dynamics state update | 3.59 | 0.69 | 4.62 | 12.46 | 278.45 | 5206 |
| Safety assessment | Collision assessment | 0.30 | 0.09 | 0.44 | 2.60 | 3321.62 | 2064 |
| Robot speed command | Speed-ratio command call | 1.39 | 0.32 | 1.89 | 5.05 | 721.26 | 2064 |

# Table 2. Online candidate-pair and command latency

| Metric | Mean [ms] | P95 [ms] | Max [ms] | N |
| --- | ---: | ---: | ---: | ---: |
| Human-M_eff query latency | 19.46 | 25.87 | 43.63 | 2064 |
| Robot-M_eff query latency | 2.64 | 3.87 | 40.16 | 2064 |
| Candidate-pair assessment latency | 25.18 | 30.09 | 49.20 | 2064 |
| Speed-command call latency | 1.39 | 1.89 | 5.05 | 2064 |
| End-to-end assessment-to-command latency | 26.84 | 32.12 | 53.72 | 2064 |
