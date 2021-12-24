# node clustering
Individual project

# 1. Justification of why you have chosen your topic.
The topic is associated with the uav trajectory problem which is typically solved through two steps: clustering nodes and planning the shortest path covering the clusters.  
There is an assumption that it is desired that the number of cluster should be as small as possible.
# 2. What is the topic?
The topic is grouping K nodes into N clusters, where N is no more than a given integer number, M. This can be coverted into a set covering problem.  
The nodes can be grouped into cluster if they all locate within radius r of the point with the XY-coordinate that is the geometric mean of those nodes' XY-coordinates.  
# 3. Design decision explaining why you select:
## 3.1. Parameters such as the size of an initial population.

The number of the nodes, K: 100;  
The required maximum number of clusters, M: 50;  
The ratio of the elitist to the next generation population, alpha: 0.5
The size of population, P: [1000, inif), 1000 (R: Larger population can acelerate the convergence of the algorithm at the expense of time cost);  
The probability of mutation: [0, 0.5], 0.2 (R: If more than 0.5, the algorithm is the same as the random strategy);  
The round of crossover: [P/10, P], P (R: More offsprings can acelerate the convergence of the algorithm at the expense of time cost);  
The radius, r: 100 (R: it is not determined yet)  
The maximum interval interval between two generation where best-so-far result occurs, Max_inter: 500.  
The input file name: "./centroid_blobs_ILP.txt"  
The output file name: "./centers_second.txt" 

The individual: an array with the size of K  
The individual example (5 nodes):
|node ID|1|2|3|4|5|
|-|-|-|-|-|-|
|label|0|0|1|0|1|  

if label is 1, the corresponding node is the center of a cluster
## 3.2. Stopping criteria.
The interval between two generation where best-so-far result occurs is more than Max_inter.
## 3.3. Fitness function.
![image](https://github.com/meihaoran/Advanced-Software-Analysis-202102/blob/main/fitness2.png
M is the threshold on the number of cluster. Noted that the number of generated cluster should be less than M.  
error is the number of nodes which are not covered by any cluster.  
O is the amount of overlaps.  
C is the set of cluster center.  
Node is the set of nodes.  
r is the redius of the circle shaped cluster.  
1{.} is indicate function  
|.| is the size of a set.  
||a,b|| is the euclidean distance between the point a and b.
## 3.4. Selection operator.
tournament selection
## 3.5. Crossover operator.
Double points crossover  
## 3.6. Mutation operator.
Single point mutation: randomly select two positions on the individual and swap their values.  
## 3.7. Generational selection strategy.
Select top-P best individuals including offsprings in the current generation as the population of the next generation.
# 4. How to run your project.
Load the probject using pychrom, execute the file "main.py" in the SourceCode.  
Please check whether you have installed the packages: numpy, scipy and matplotlib before executing.  
# 5. How to adjust parameters.
The variables from line 15 to line 24 in "main.py" correspond to the mentioned parameters.  
You could adjust all variables except K which should be associated with the input file and fileName.  

