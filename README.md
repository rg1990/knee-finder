# knee-finder
This repository implements the algorithm described here: https://www.mdpi.com/1996-1073/14/4/1206, used to identify the location of knee points and knee onsets in battery capacity degradation curves. The paper describes internal resistance "elbows", however the algorithm and this implementation can be applied to both knees and elbows.

<h3>Usage</h3> 
Creating an instance of KneeFinder only requires x and y data arrays. The analysis is performed upon instantiation.<br>


```python
# Use default settings
kf = KneeFinder(x, y)
``` 

You can optionally enable automatic truncation for handling sigmoid-like curves. Valid options for <code>mode</code> are <code>'knee'</code> and <code>'elbow'</code>.<br>


```python
# Other optional settings
kf = KneeFinder(x, y, automatic_truncation=False, mode='knee')
```

The results are stored in the attributes <code>onset</code>, <code>point</code>, <code>onset_y</code>, <code>point_y</code>, <code>eol_cycle</code>, and <code>eol_val</code>.


```python
# Print x results (cycle numbers)
print(np.round(kf.onset, 2))
>> 280.76
print(np.round(kf.point, 2))
>> 344.69

# Print y results (capacity values)
print(np.round(kf.onset_y, 2))
>> 1.04
print(np.round(kf.point_y, 2))
>> 1.01

# Print the end of life info
print(np.round(kf.eol_cycle, 2))
>> 447
print(np.round(kf.eol_val, 2))
>> 0.86
```

You can plot the results using <code>kf.plot_results</code>. This method has optional parameters to show different curve fits (monotonic, line-plus-exponential, asymmetric sigmoid.) The <code>data_style</code> parameter controls the line style for the data.

```python
kf.plot_results(mon=False, line_exp=False, sig=False, data_style='-')
```

<img src="img/severson_example_result.png" width="640">




---
<h3>Try it Yourself</h3> 
You can play with KneeFinder using the interactive Streamlit app available here: https://rg1990-knee-finder-streamlit-knee-finder-mwgskp.streamlit.app/. In the app, you can see the results using example data from 120 cells (Severson et al. [2]), a fake sigmoidal degradation curve, or you can upload your own data in CSV format.


---
<h3>Acknolwedgements & References</h3>
The application of Bacon-Watts and double Bacon-Watts to locate the knee point and onset was described by Fermín-Cueto et al. in [1].<br><br>

The data used in this repo and in the associated Streamlit app is taken from [2].<br>

The method was extended by me (Richard Gilchrist), Calum Strange, Shawn Li and Goncalo dos Reis, as described in Strange et al. [3]. KneeFinder implementation by Richard Gilchrist.

<br><br>

[1] P. Fermín-Cueto et al., "Identification and machine learning prediction of knee-point and knee-onset in capacity degradation curves of lithium-ion cells," Energy and AI, vol. 1, p. 100006, 2020. doi:10.1016/j.egyai.2020.100006

[2] Severson, K.A., Attia, P.M., Jin, N. et al. "Data-driven prediction of battery cycle life before capacity degradation." Nat Energy 4, 383–391 (2019). doi: 10.1038/s41560-019-0356-8

[3] C. Strange, S. Li, R. Gilchrist, and G. dos Reis, "Elbows of internal resistance rise curves in li-ion cells," Energies, vol. 14, no. 4, p. 1206, 2021. doi:10.3390/en14041206
