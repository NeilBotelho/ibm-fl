1. Download the cancer data into the research directory in a folder called source_data
its structure should be as follows:
```
research
|
----source_data
		|
		-----train
		|		|
		|		-----malignant
		|		|       0.jpg.....
		|		-----benign
		|				10.jpg
		|
		-------test
				same as above
```


2. With current directory as top of the ibmfl repo, run:
```python  examples/generate_configs.py -n 2 -m keras_classifier -d mnist -p examples/data/mnist/random/```

to create data for each party. You can customise number of parties and number of data points per party as well. Use 
```python  examples/generate_configs.py -h```
for a list of all customisation options

3. Similarly run 
```python examples/generate_configs.py -n 2 -m keras_classifier -d mnist -p examples/data/mnist/random/```
to generate the required configuration files

4. From here you can basically follow the quick start guide in the root of the repository but substitute ```examples``` with ```research```


