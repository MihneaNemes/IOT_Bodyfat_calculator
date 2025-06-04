Body Fat Predictor Project Documentation
Overview
The Body Fat Predictor project is a comprehensive system designed to estimate body fat percentage using images of a person's front and side profiles, combined with height and weight inputs. The system leverages deep learning for image processing and body fat prediction, integrates with AWS for data storage, and provides a user-friendly interface for capturing images and viewing results. The project includes a desktop application built with Tkinter, a web application using Flask, and a machine learning model trained on the BodyM dataset.
The system captures images using a Raspberry Pi camera (or compatible hardware), processes them with a pre-trained deep learning model, stores results in AWS (S3 and DynamoDB), and visualizes historical predictions via a web interface. The machine learning model is a ResNet50-based regressor trained to predict body fat percentage, with image preprocessing enhanced by DeepLabV3 for person segmentation.
Project Components
The project consists of four main Python scripts and one HTML template, each serving a specific role:
1.	app.py: A Tkinter-based desktop application for capturing images, running predictions, and storing results in AWS.
2.	predict_body_fat.py: Contains the logic for loading the trained model, processing images, and predicting body fat percentage.
3.	web_app.py: A Flask-based web application for viewing past predictions stored in AWS DynamoDB.
4.	index.html: The HTML template for the Flask web application, displaying predictions and charts.
5.	train_model.py: Handles the training of the body fat prediction model using the BodyM dataset.
System Requirements
Hardware
•	Raspberry Pi (or compatible device) with a camera module for image capture (used in app.py).
•	A computer for running the training script (train_model.py) and hosting the web application (web_app.py).
Software
•	Python 3.8+
•	Libraries:
o	tkinter for the GUI in app.py.
o	Pillow (PIL) for image processing.
o	picamera2 for Raspberry Pi camera control.
o	boto3 for AWS integration (S3 and DynamoDB).
o	torch and torchvision for deep learning model operations.
o	opencv-python (cv2) for image processing.
o	numpy, pandas, matplotlib for data handling and visualization.
o	flask for the web application.
o	tqdm for progress bars during training.
o	scikit-learn for evaluation metrics in training.
•	AWS Account: Configured with access to S3 and DynamoDB services.
•	BodyM Dataset: Used for training the model, containing images and metadata (height, weight, gender, etc.).
AWS Configuration
•	S3 Bucket: Named bodyfat-predictor-images for storing front and side images.
•	DynamoDB Table: Named BodyFatPredictions with prediction_id as the partition key.
•	Ensure AWS credentials are configured (e.g., via ~/.aws/credentials or environment variables).
Requirements.txt:
torch
torchvision
pillow
opencv-python
numpy
pandas
matplotlib
flask
boto3
picamera2
tqdm
scikit-learn
1.	Set Up AWS Credentials:
Configure AWS credentials using the AWS CLI:
2.	aws configure
Provide your AWS Access Key, Secret Access Key, and region.
3.	Prepare the BodyM Dataset:
o	Ensure the dataset includes:
	train, testA, and testB directories.
	measurements.csv, hwg_metadata.csv, and subject_to_photo_map.csv files.
	Image directories (mask or mask_left).
4.	Train the Model (optional if using pre-trained model):
Run train_model.py to train the model and generate best_bodyfat_regressor_model.pth and bodyfat_norm_params.npz:
5.	python train_model.py
Usage
1. Training the Model (train_model.py)
•	Purpose: Trains a ResNet50-based regression model to predict body fat percentage using the BodyM dataset.
•	Key Features:
o	Uses DeepLabV3 for person segmentation to isolate the subject in images.
o	Applies data augmentation (random crops, flips, rotations) for training.
o	Normalizes body fat targets using mean and standard deviation.
o	Splits data into training (80%) and validation (20%) sets, with optional testing on testA and testB.
o	Saves the best model (best_bodyfat_regressor_model.pth) and normalization parameters (bodyfat_norm_params.npz).
•	Execution:
•	python train_model.py
•	Output:
o	Trained model saved to bodym_dataset/training_outputs/best_bodyfat_regressor_model.pth.
o	Normalization parameters saved to bodym_dataset/training_outputs/bodyfat_norm_params.npz.
o	Console output includes training/validation loss, learning rate, and test set metrics (MAE, RMSE, R²).
2. Running the Desktop Application (app.py)
•	Purpose: Captures front and side images, predicts body fat percentage, and stores results in AWS.
•	Key Features:
o	GUI built with Tkinter for user input (name, height, weight) and image capture.
o	Uses picamera2 to capture images with a 3-second delay.
o	Integrates with predict_body_fat.py for predictions.
o	Uploads images to S3 and saves prediction data to DynamoDB.
o	Displays past predictions in a table with an option to view associated images.
•	Execution:
•	python app.py
•	Usage Steps:
1.	Launch the application to open the Tkinter GUI.
2.	Select a name from the dropdown or enter a new one.
3.	Enter height (cm) and weight (kg).
4.	Click "Capture Front View" and "Capture Side View" to take images (3-second countdown).
5.	Click "Run Prediction" to estimate body fat percentage and category (e.g., Essential Fat, Athletic).
6.	Click "Save to AWS" to store the results and images.
7.	Click "View Past Predictions" to see a table of previous results and view associated images.
•	Notes:
o	Images are temporarily saved in /tmp/bodyfat_predictor and deleted on application exit.
o	The camera is initialized with a retry mechanism (up to 3 attempts).
o	Requires a working Raspberry Pi camera module.
3. Running the Web Application (web_app.py and index.html)
•	Purpose: Provides a web interface to view past predictions stored in DynamoDB.
•	Key Features:
o	Displays a table of predictions grouped by name.
o	Includes a dropdown to filter predictions by name.
o	Generates a chart of body fat percentage over time (requires Chart.js, not fully implemented in the provided index.html).
•	Execution:
•	python web_app.py
•	Access:
o	Open a browser and navigate to http://localhost:5000.
•	Notes:
o	The provided index.html is incomplete and requires Chart.js integration for the chart.
o	Predictions are sorted by timestamp for each user.
4. Predicting Body Fat (predict_body_fat.py)
•	Purpose: Processes images and predicts body fat percentage using the trained model.
•	Key Features:
o	Loads the pre-trained ResNet50 model and normalization parameters.
o	Uses DeepLabV3 for person segmentation to isolate the subject.
o	Processes front and side images, averaging their predictions.
o	Classifies body fat percentage into categories (Essential Fat, Athletic, Fitness, Average, Obese).
o	Visualizes original, segmented, and processed images using Matplotlib.
•	Execution (standalone):
•	python predict_body_fat.py
•	Notes:
o	Integrated into app.py for the desktop application.
o	Requires model and normalization parameter files in the specified paths.
o	Height and weight inputs are logged but not used in the prediction.
Data Flow
1.	Image Capture (app.py):
o	User captures front and side images via the Tkinter GUI.
o	Images are saved temporarily in /tmp/bodyfat_predictor.
2.	Prediction (predict_body_fat.py):
o	Images are segmented using DeepLabV3 to isolate the person.
o	Processed images are fed into the ResNet50 model for body fat prediction.
o	Predictions are averaged and categorized.
3.	Storage (app.py):
o	Images are uploaded to S3 with keys like images/<prediction_id>_front.jpg.
o	Prediction data (ID, name, height, weight, body fat, category, timestamp) is stored in DynamoDB.
4.	Visualization (web_app.py):
o	Retrieves predictions from DynamoDB.
o	Displays them in a table and (optionally) a chart in the web interface.
Model Training Details
•	Dataset: BodyM dataset, containing:
o	Images of subjects (front or side views) in mask or mask_left directories.
o	Metadata files (measurements.csv, hwg_metadata.csv, subject_to_photo_map.csv) with subject IDs, height, weight, gender, and other measurements.
•	Preprocessing:
o	Images are segmented using DeepLabV3 to isolate the person.
o	Augmentations (random crops, flips, rotations) are applied during training.
o	Body fat targets are normalized using mean and standard deviation.
•	Model Architecture:
o	Base: Pre-trained ResNet50 (ImageNet weights).
o	Modifications: Replaced fully connected layer with a regression head (2048→512→128→1).
o	Loss: Huber loss for robust regression.
o	Optimizer: AdamW with learning rate 5e-5 and weight decay 1e-2.
o	Scheduler: ReduceLROnPlateau (factor=0.2, patience=5).
•	Training:
o	Epochs: Up to 50, with early stopping after 10 epochs without improvement.
o	Batch size: 8 (optimized for CPU).
o	Splits: 80% train, 20% validation, with optional testing on testA and testB.
•	Evaluation:
o	Metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R² score.
o	Results are printed for the test set (if available).
AWS Integration
•	S3:
o	Stores front and side images with unique keys based on prediction_id.
o	Example key: images/<uuid>_front.jpg.
•	DynamoDB:
o	Table: BodyFatPredictions.
o	Attributes: prediction_id (string), name (string), height_cm (decimal), weight_kg (decimal), predicted_body_fat (decimal), category (string), front_image_key (string), side_image_key (string), timestamp (string).
o	Used for storing and retrieving prediction data.
Limitations
•	Camera Dependency: The desktop application requires a Raspberry Pi camera module.
•	Incomplete Web Interface: The index.html template lacks Chart.js integration for visualizing body fat trends.
•	Dataset Dependency: The model relies on the BodyM dataset, which may not be publicly available.
•	CPU Usage: The model is configured for CPU to ensure compatibility, which may limit performance on large datasets.
•	Height/Weight Usage: These inputs are collected but not used in the prediction model, limiting their utility.
•	Gender Handling: The training script uses wrist measurements as a proxy for neck circumference, which may introduce errors.
License
This project is intended for educational and research purposes. The BodyM dataset and any pre-trained models may have specific licensing requirements; ensure compliance with their terms.

