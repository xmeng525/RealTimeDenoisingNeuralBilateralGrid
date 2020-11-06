## Outlier Removal Example
We apply a preprocessing step to detectand suppress outlier pixels for the 64spp radiance. The algorithm is adapted from Section 4.3 in [LBF15](https://web.ece.ucsb.edu/~psen/Papers/SIGGRAPH15_LBF_LoRes.pdf). However, we apply one modification compared to the previous work: the weights depend not only on Euclidean distance of pixels, but also on the radiometric differences. We use albedo & shading normals to test the radiometric difference. 

Please checkout the code and the figure links for more detail.

[Example of Input](https://drive.google.com/file/d/18Q1mswv5NFOZhVwlVFHZvhTDAyJupTrH/view?usp=sharing)

[Example of Output](https://drive.google.com/file/d/1C3GJlC6OkPvUpga3LIfKdBHoB6MD-Yik/view?usp=sharing)
