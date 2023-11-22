# Hepatocellular Carcinoma Segmentation Project

## Project Description

This project aims to develop an automatic detection and segmentation deep learning model for Hepatocellular Carcinoma (HCC) using Dynamic Contrast Enhanced (DCE) MRI.

## Usage

To use this project, you need to:

1. Prepare your data in the required format according to nnU-Net toolbox [1], as in `datalist/five_fold_dict.json`,.
2. Adjust the parameters in the training script as needed.
3. Run the training script.
4. Use the trained model to make predictions on new data.

## Dependencies

This project requires the following dependencies:

- python 3.8.16 or later
- torch 1.0 or later
- monai
- aim
- tqdm

## Footscripts

```bash
echo train model with data in fold 0
python training.py 0
```

```bash
echo predict using trained model of fold 0
python predict.py INPUT_DIRECTORY_PATH OUTPUT_DIRECTORY_PATH 0
```

## License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.

## Contact

For any questions or concerns, please open an issue on GitHub.