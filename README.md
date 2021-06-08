# 2D Kalman Filter for Obstacle Tracking and Prediction for Autonomous Surface Vehicles 

## Author: Luca Chun Lun Lit

This research was conducted as a Mathematics undergradute culminating experience (MATH 87) under the supervision of Professor Alberto Quattrini Li, Professor of Computer Science at Dartmouth College, and Professor Feng Fu, Professor of Mathematics at Dartmouth College. I have worked closely and received a great deal of guidance from both Professors, as well as Ph.D. student Mingi Jeong. 

### Description: 
This codebase generates a simulation of multiple-obstacle tracking and prediction using the 2D Kalman Filter. Applying this methodology to Professor Li's Autonomous Surface Vehicle, the ASV will be able to compute trajectories to avoid obstacle collisions. 

**Current state:** handling linear + non-linear motion with acceleration and angular velocity

### Important files:
```
├── 2D-Kalman-Filter			
│   ├── KalmanFilter.py
│   ├── objTrackingv3_multiobs.py
│   ├── object.py
```

## Simulation:
- The following simulation shows a side by side plot of 2 obstacles A, B and the Autonomous Surface Vehicle (ASV) moving in the same locality.
- The left plot shows the actual trajectory of the obstacles 
- The right plot shows the measurement and Kalman Filter predicted trajectories of the obstacles

https://user-images.githubusercontent.com/47261209/121256096-34109780-c87a-11eb-90fd-d09c64464d07.mov


### References: 
This code was inspired by the following article: 
https://machinelearningspace.com/object-tracking-2-d-object-tracking-using-kalman-filter-in-python/



