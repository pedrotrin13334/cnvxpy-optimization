## Usage 
Use **pip install -r requirements.txt** to add the necessary dependencies to make the python script run, then do:
```console
foo@bar:~$ python ./battery_problem NUMBER_OF_BINS 
```

optionally you can add a float value to the weight parameter for the stdev as the second argument.

## The convex Optimization problem 

$$
\begin{align*}
& \max_{x_i,j} & & \min\{\sum_j x_{i,j}v_j\} -w \times \text{stdev}(\{\sum_j x_{i,j}v_j\})  \nonumber \\
& \textrm{s.t.} & & \sum_{i} x_{i,j} = 4,\\
& & &  \sum_{j} x_{i,j} \leq n_j\nonumber, \\
& & &  x_{i,j} \geq 0\nonumber. \\
\end{align*} $$