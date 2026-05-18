# training and tuning architecture planning doc
finding the parameters for the kalman filter are super tedious and tiring. it appears that vendor spec data that should work (like measurement covariances based on the imu noise parameters, and initial covariances based on sitting parameters) ends up not working / not converging or even tracking, and only god knows why. instead, i've designed a way to "learn" the best parameters
- select parameters to tune, typically the initial covariances for position, velocity, quat, omega, both biases, zvup threshold, and the zvup noise matrix diagonals
- take some known trials from the vicon data with known initial positions
- for each trial, run the c++ filter iteratively using some chosen parameters selected via grid search
- score the filter performance vs ground truth on the trial by using MSE + terminal cost
- average trial performance into composite score
- feed this into grid search so it can adapt
- repeat until score good

this is.. okay, and produced good-ish performance on the trials (it's not traditional machine learning and i have no idea if it's overfitting or actually performing well) but there's a lot of room for improvement
- running the grid search optimizer is very very slow, and it usually takes upwards of 2000 trials to see good performance
- performance has been capped at something like 30cm linearly terminal weighted MSE (so an unintelligible benchmark, since it's scaled averaged mean square error, but it's useful as a cost function for keeping the filter prioritizing where the guy ends up) and i cannot figure out how to break the benchmark, but it's clearly not good enough, since on some trials you'll see some inaccuracies
- i'm currently using the zero velocity update provided by the ground truth data, but this is not feasible for the actual prototype. I also really don't want to run the thousands of trials required to get this to work with the real zvup
- I'm not sure which parameters affect the filter performance the most, and aggregating the data has caused weird effects (the treadmill data in particular needs to be set up in a way where it is standing velocity normalized, otherwise it skews the training due to the zero velocity assumption on treadmill walking being false, so right now it cannot even be used, pshtt)
- running the filter on the cpp is slow. it's there to ensure consistency between the python analysis and the real code when it's time to deploy, but for training and tuning purposes i should probably have an interface and a pure python implementation that is completely vectorized
- For whatever god forsaken reason, every time i try to add a secondary measurement beyond just zero velocity update, like quaternion measurements from a better onboard orientation only filter or magnetometer measurements from the imu, performance always degrades, and tuning is so slow that it's hard to benchmark whether an improvement is made by using such a thing
- I need to figure out the best performer from the permutation of zvup measurement (threshold too), magnetometer (threshold) or no magnetometer, quaternion (threshold) or no quaternion, maybe even foot force sensor vs no foot force sensor, there's a lot, sigh
- quaternion tracking always degrades with zvup enabled. I have no idea what is causing this.
- I need some better visualization or other insight into what is happening in the covariance matrix
- no idea if the zvup lever arm model is required or not, but not having the zvup positioned at the foot is gonna hurt us, a lot. I have no idea what is happening at the ankle-to-foot transformation tbh


# possible thoughts for improvement
- k fold cross validation on tests, tho idk how i'd do this well
- modularizing the testing and tuning harness, it's kind of bloated right now and will make implementing larger and more thorough testing, like using crossval or testing every possible permutation of sensor updates, etc etc, kind of difficult
- raylib so i can run the hyperparameter optimization massively parallel
- better cost function?
- using a real machine learning algorithm or some gradient nonsense to find the parameters (kalman filters are maybe not differentiable?)
- redesigning the whole position filter to use the quaternion output by the onboard filter
