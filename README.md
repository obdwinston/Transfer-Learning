<div align="center">
	<img src="https://github.com/obdwinston/Transfer-Learning/assets/104728656/f7cc324f-a3fe-4e1b-89bb-c45f718bcc88">
</div>

A simple PyTorch transfer learning tutorial with VGG-16 as feature extractor. The classifier predicts **first generation** Pokemon with about 95% test accuracy.

1. Clone the repository:
```
git clone https://github.com/obdwinston/Transfer-Learning.git
```
2. Download and extract [Pokemon dataset from Kaggle](https://www.kaggle.com/datasets/lantian773030/pokemonclassification) to root folder.
3. Run the shell script from the root folder:
```
./run.sh
```
Note:
- Check that you have the required packages installed. See `requirements.txt`.  
- If an error is thrown regarding execution permission, run the following in the terminal: `chmod +x run.sh`.
- Model training is required for the **first run**, after which the model is saved (`model.pt` in `src` folder) and you can comment out `python3 src/utils.py` and `python3 src/train.py` in `run.sh`.
- After the first run, you can also choose to run only the predict script from the root folder (replace \<url\> with desired image URL):
```
python3 src/predict.py "<url>"
```
