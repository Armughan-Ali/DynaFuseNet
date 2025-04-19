# SignPoseFusion

A hybrid deep learning pipeline for sign language recognition using image frames and 3D skeleton pose data.

## 📁 Folder Structure



## 🚀 Installation

1. **Clone the repo:**
```bash
git clone https://github.com/yourusername/signposefusion.git
cd signposefusion
Create Anaconda Environment:

bash
Copy
Edit
conda create -n signposefusion python=3.9 -y
conda activate signposefusion
Install Dependencies from requirements.txt:

bash
Copy
Edit
pip install -r requirements.txt
Alternatively, you can manually install the necessary packages:

bash
Copy
Edit
pip install torch torchvision opencv-python numpy scikit-learn
📦 requirements.txt
Make sure to include the following dependencies in your requirements.txt file:

ini
Copy
Edit
torch==1.10.0
torchvision==0.11.1
opencv-python==4.5.3.56
numpy==1.21.2
scikit-learn==0.24.2
📊 Dataset Format
images/: Frame-wise RGB images.

skeleton.npz: Skeleton data in numpy format.

python
Copy
Edit
{
    'keypoints': [...],  # shape (N, 50)
    'labels': [...]      # shape (N,)
}
🏋️‍♂️ Training
Run the following command to start training:

bash
Copy
Edit
python train.py
Model Output
The trained model will be saved in the outputs/ directory as model.pt.