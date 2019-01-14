# deep
Deep learning using C++

This is under construction for eventual use in real time data.

Objective:
This project aims to develop deep learning using ANN 
and backpropagation algorithm.                       
                                                      
Author: Golam Gause Jaman                            
email: jamagola@isu.edu                              

 
If you want to use an array, you'll have to allocate a new block, large enough for the whole new array, 
and copy the existing elements across before deleting the old array. Or you could use a more complicated 
data structure that doesn't store its elements contiguously.
Luckily, the standard library has containers to handle this automatically; including vector, a resizable array.

e.g. std::vector<float> X;
 
Source: https://stackoverflow.com/questions/29015546/allocate-more-memory-for-dynamically-allocated-array

Followings are some key information:
  ->Pattern by pattern learning when update occurs for each data pair.
  ->Batch or mini-batch learning when update occurs after group of data pairs.
  ->Layers include input, hidden and output. Minimum layers supported is 3.
  ->Learning rate: Not adaptive (needs attention).
  ->Bias: Single constant used across all nodes with variable weight.
  ->Trained using same sequence a provided 
  ->(needs attention, use of n-fold, n segmented data and use one for verification , rest for training).
  ->Stopping criteria: Epoch only (needs attention, error index).
  ->Error: Mean Square Error.
  ->Output range from each node: 0 ~ 1
  ->Activation function used: Sigmoid
