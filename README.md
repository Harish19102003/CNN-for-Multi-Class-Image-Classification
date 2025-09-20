# CNN-for-Multi-Class-Image-Classification

A Convolutional Neural Network (CNN) built with PyTorch to classify images into **Cats, Dogs, and Snakes**.

## 🗂️ Dataset Structure

Organize your dataset as follows:
```
data/
  Animals/
    cats/
    dogs/
    snakes/
```

## 🔧 Setup

### Local Development

```bash
git clone https://github.com/Harish19102003/CNN-for-Multi-Class-Image-Classification.git
cd CNN-for-Multi-Class-Image-Classification
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix/Mac:
source venv/bin/activate
pip install -r requirements.txt
```

## 🚀 Usage

### 1. Prepare Dataset

Place your images in the `data/Animals` folders as shown above.

### 2. Train the Model

```bash
python train.py --data_dir data/Animals --epochs 10 --batch_size 32
```

### 3. Evaluate the Model

```bash
python evaluate.py --data_dir data/Animals --model_path saved_model.pth
```

### 4. Predict on New Images

```bash
python predict.py --image_path path/to/image.jpg --model_path saved_model.pth
```

## 🏗️ Project Structure

```
CNN-for-Multi-Class-Image-Classification/
│
├── train.py
├── evaluate.py
├── predict.py
├── model.py
├── requirements.txt
├── README.md
└── data/
    └── Animals/
        ├── cats/
        ├── dogs/
        └── snakes/
```

## 📦 Requirements

- Python 3.7+
- PyTorch
- torchvision
- numpy
- matplotlib

Install all dependencies with:
```bash
pip install -r requirements.txt
```

## 📈 Results

Add your training accuracy, loss curves, and sample predictions here.

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License.