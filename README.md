Introduction
The Baker Hughes Challenge presents a common issue in modern-day algorithmic modeling: creating
accurate data representations given limited computational power. With a mission of improving energy in
an efficient and safe manner, having accurate models that better predict the lifespan of products is critical
for maintenance and repair procedures. The following documentation outlines the process of preparing
model training for an electric motor.

Background
The dataset provided gives 500,000 points that each have a corresponding frequency, power, and vibration
level. The variable-frequency drive determines a Hertz value which corresponds to the input rate of
electricity. The power is a measurement of watts of the engine’s output and the vibration level is the value
to be forecasted by the model. Both of these units are normalized to one another so that they can be
graphed effectively. The Baker Hughes Challenge requires downsampling, reducing the dataset to 2,500
data points that model the data in three ways:

1) Density-Based Selection - Identify 200 different clusters and find the best match within each
cluster to be selected among the 2,500 data points. The size of the clusters should vary depending
on the density of the frequency-power pair. Areas with high data density should be represented
with smaller cluster sizes while areas with lower data density should be represented with large
clusters. In a broader context, the overall distribution of the data should match that of the original
dataset only but with 0.5% of the size.

2) Uniform Selection - The uniform distribution selection model should produce a similar output as
the density-based selection in terms of the range of values covered, but should have a more even
distribution. The cluster size across different densities should remain relatively constant to
provide a different data distribution.

3) Hybrid Selection - The hybrid selection provides an in-between solution set of 2,500 data points.
Different parameters of the hybrid selection algorithm should outline a distribution that is affected
by data density, but to a lesser extent than the density-based selection.
The purpose of creating three different datasets is to diversify the output of the prediction models. While
in testing, the model is trained on multiple datasets and monitored for accuracy. Because of the unique
performance of specific products, in real production environments, certain models may be a more
accurate fit in different stages of the product’s life. By providing multiple datasets, the likelihood of
creating an accurate prediction model is substantially increased.

More information can be found in the complete documentation linked here:
https://drive.google.com/file/d/1JMLWMsQDiDozFRWFXhn752RjBwWNU_9C/view?usp=sharing

The Google Drive folder containing the CSV files, data visualizations, and project presentation can be found here:
https://drive.google.com/drive/folders/1I0dAvUnOGW33BLSLVn-N_0BPp0K12vEa?usp=drive_link
