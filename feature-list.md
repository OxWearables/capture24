# Hand Crafted Features List
The table below presents an overview of the hand crafted features that have been curated as inputs for the Random Forest and XGBoost models. 
These features are extracted from the Euclidean norm of the triaxial accelerometer data, minus 1 to remove gravity.

| Feature Name                    | Description                                                 | Units           |
|---------------------------------|-------------------------------------------------------------|-----------------|
| <b>Moment features</b>                                                                                          |
| avg                             | Mean                                                        | g               |
| std                             | Standard deviation                                          | g               |
| skew                            | Skewness                                                    |                 |
| kurt                            | Kurtosis                                                    |                 |
| <b>Quantile Features</b>                                                                                        |
| min                             | Minimum                                                     | g               |
| q25                             | Lower quartile                                              | g               |
| med                             | Median                                                      | g               |
| q75                             | Upper quartile                                              | g               |
| max                             | Maximum                                                     | g               |
| <b>Autocorrelation features</b>                                                                                 |
| acf_1st_max                     | Maximum autocorrelation                                     |                 |
| acf_1st_max_loc                 | Location of 1st autocorrelation maximum                     | s               |
| acf_1st_min                     | Minimum autocorrelation                                     |                 |
| acf_1st_min_loc                 | Location of 1st autocorrelation minimum                     | s               |
| acf_zeros                       | Number of autocorrelation zero-crossings                    |                 |
| <b>Spectral features</b>                                                                                        |
| pentropy                        | Signal's spectral entropy                                   | nats            |
| power                           | Signal's total power                                        | g<sup>2</sup>/s |
| f1, f2, f3                      | 1st, 2nd and 3rd dominant frequencies                       | Hz              |
| p1, p2, p3                      | Power spectral densities of respective dominant frequencies | g<sup>2</sup>/s |
| fft0, fft1, fft2, ...           | Power spectral density for frequencies 0Hz, 1Hz, 2Hz, ...   | g<sup>2</sup>/s |
| <b>Peak features</b>                                                                                            |
| npeaks                          | Number of peaks in the signal per second                    | 1/s             |
| peaks_avg_promin                | Average prominence of peaks                                 | g               |
| peaks_min_promin                | Minimum prominence of peaks                                 | g               |
| peaks_max_promin                | Maximum prominence of peaks                                 | g               |
