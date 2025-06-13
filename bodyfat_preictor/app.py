import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
from picamera2 import Picamera2
import threading
import time
import os
import boto3
import uuid
from io import BytesIO
from datetime import datetime
import sys
from decimal import Decimal
import cv2

# Import the predict_body_fat function from the provided script
from predict_body_fat import predict_body_fat

model_path = "best_bodyfat_regressor_model.pth"
norm_params_path = "bodyfat_norm_params.npz"

class BodyFatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Body Fat Predictor")
        self.camera_lock = threading.Lock()

        # Initialize AWS clients
        self.s3_client = boto3.client('s3')
        self.dynamodb = boto3.resource('dynamodb')
        self.bucket_name = "bodyfat-predictor-images"
        self.table_name = "BodyFatPredictions"
        self.table = self.dynamodb.Table(self.table_name)

        # Initialize the camera with retry mechanism
        self.camera = None
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.camera = Picamera2()
                self.camera.configure(self.camera.create_still_configuration())
                break
            except Exception as e:
                print(f"Camera init attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    messagebox.showerror("Camera Error", f"Failed to initialize camera after {max_retries} attempts: {str(e)}")
                    self.close_app()
                time.sleep(1)

        # GUI Elements
        tk.Label(root, text="Name:").grid(row=0, column=0, sticky="e")
        tk.Label(root, text="Height (cm):").grid(row=1, column=0, sticky="e")
        tk.Label(root, text="Weight (kg):").grid(row=2, column=0, sticky="e")

        # Name dropdown
        self.names = ["John", "Jane", "Alex", "Sarah"]  # Predefined names; can be modified
        self.name_var = tk.StringVar()
        self.name_dropdown = ttk.Combobox(root, textvariable=self.name_var, values=self.names)
        self.name_dropdown.grid(row=0, column=1)
        self.name_dropdown.set("Select a name")

        self.height_entry = tk.Entry(root)
        self.weight_entry = tk.Entry(root)
        self.height_entry.grid(row=1, column=1)
        self.weight_entry.grid(row=2, column=1)

        self.preview_label = tk.Label(root)
        self.preview_label.grid(row=3, column=0, columnspan=2, pady=10)

        self.capture_front_btn = tk.Button(root, text="Capture Front View", command=lambda: self.capture("front"))
        self.capture_front_btn.grid(row=4, column=0, columnspan=2, pady=5)

        self.capture_side_btn = tk.Button(root, text="Capture Side View", command=lambda: self.capture("side"))
        self.capture_side_btn.grid(row=5, column=0, columnspan=2, pady=5)

        self.predict_btn = tk.Button(root, text="Run Prediction", command=self.run_prediction)
        self.predict_btn.grid(row=6, column=0, columnspan=2, pady=10)

        self.save_btn = tk.Button(root, text="Save to AWS", command=self.save_to_aws, state="disabled")
        self.save_btn.grid(row=7, column=0, columnspan=2, pady=5)

        self.view_btn = tk.Button(root, text="View Past Predictions", command=self.view_past_predictions)
        self.view_btn.grid(row=8, column=0, columnspan=2, pady=5)

        self.result_label = tk.Label(root, text="", font=("Arial", 12))
        self.result_label.grid(row=9, column=0, columnspan=2)

        self.captured_images = {}
        self.image_paths = {"front": None, "side": None}
        self.prediction_result = None
        self.prediction_id = None

        # Temporary directory for saving captured images
        self.temp_dir = "/tmp/bodyfat_predictor"
        os.makedirs(self.temp_dir, exist_ok=True)

        self.root.protocol("WM_DELETE_WINDOW", self.close_app)

    def capture(self, view):
        threading.Thread(target=self._delayed_capture, args=(view,), daemon=True).start()

    def _delayed_capture(self, view):
        with self.camera_lock:
            self.result_label.config(text=f"{view.capitalize()} capture in 3 seconds...")
            time.sleep(3)
            try:
                if self.camera is None:
                    raise RuntimeError("Camera not initialized")
                self.camera.start()
                time.sleep(2)
                image = self.camera.capture_array()
                self.camera.stop()
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)
                self.captured_images[view] = pil_image
                image_path = os.path.join(self.temp_dir, f"{view}_{uuid.uuid4()}.png")
                pil_image.save(image_path)
                self.image_paths[view] = image_path
                img_tk = ImageTk.PhotoImage(pil_image.resize((320, 240)))
                self.preview_label.configure(image=img_tk)
                self.preview_label.image = img_tk
                self.result_label.config(text=f"{view.capitalize()} image captured!")
            except Exception as e:
                self.result_label.config(text=f"Capture failed: {str(e)}")
                print(f"Capture error: {e}")
                self.close_camera()

    def run_prediction(self):
        name = self.name_var.get()
        height = self.height_entry.get()
        weight = self.weight_entry.get()

        if name == "Select a name":
            messagebox.showwarning("Input Error", "Please select a name.")
            return

        if not height or not weight:
            messagebox.showwarning("Input Error", "Please enter height and weight.")
            return

        try:
            height = float(height)
            weight = float(weight)
        except ValueError:
            messagebox.showerror("Invalid Input", "Height and weight must be numbers.")
            return

        if "front" not in self.captured_images or "side" not in self.captured_images:
            messagebox.showwarning("Missing Images", "Capture both front and side images first.")
            return

        self.prediction_id = str(uuid.uuid4())
        try:
            predicted_bf, category = predict_body_fat(
                self.image_paths["front"],
                self.image_paths["side"],
                height,
                weight,
                model_path,
                norm_params_path
            )
            if predicted_bf is None or category is None:
                raise ValueError("Prediction failed.")
            self.prediction_result = (predicted_bf, category, name)
            self.result_label.config(text=f"Predicted Body Fat for {name}: {predicted_bf:.2f}% ({category})")
            self.save_btn.config(state="normal")
        except Exception as e:
            self.result_label.config(text=f"Prediction failed: {str(e)}")
            self.save_btn.config(state="disabled")
            print(f"Prediction error: {e}")

    def save_to_aws(self):
        if not self.prediction_result or not self.prediction_id:
            messagebox.showwarning("No Prediction", "Run a prediction first before saving.")
            return

        predicted_bf, category, name = self.prediction_result
        height = float(self.height_entry.get())
        weight = float(self.weight_entry.get())

        try:
            front_key = f"images/{self.prediction_id}_front.jpg"
            side_key = f"images/{self.prediction_id}_side.jpg"

            front_buffer = BytesIO()
            side_buffer = BytesIO()
            self.captured_images["front"].save(front_buffer, format="JPEG")
            self.captured_images["side"].save(side_buffer, format="JPEG")
            front_buffer.seek(0)
            side_buffer.seek(0)

            self.s3_client.upload_fileobj(front_buffer, self.bucket_name, front_key)
            self.s3_client.upload_fileobj(side_buffer, self.bucket_name, side_key)

            self.table.put_item(
                Item={
                    'prediction_id': self.prediction_id,
                    'name': name,  # Add name to DynamoDB
                    'height_cm': Decimal(str(height)),
                    'weight_kg': Decimal(str(weight)),
                    'predicted_body_fat': Decimal(str(predicted_bf)),
                    'category': category,
                    'front_image_key': front_key,
                    'side_image_key': side_key,
                    'timestamp': datetime.utcnow().isoformat()
                }
            )

            self.result_label.config(text="Data saved to AWS successfully!")
            self.save_btn.config(state="disabled")
        except Exception as e:
            self.result_label.config(text=f"Failed to save to AWS: {str(e)}")
            print(f"AWS save error: {e}")

    def view_past_predictions(self):
        try:
            view_window = tk.Toplevel(self.root)
            view_window.title("Past Predictions")
            view_window.geometry("600x400")

            response = self.table.scan()
            items = response.get('Items', [])

            if not items:
                tk.Label(view_window, text="No past predictions found.", font=("Arial", 12)).pack(pady=20)
                return

            columns = ('Prediction ID', 'Name', 'Height (cm)', 'Weight (kg)', 'Body Fat (%)', 'Category', 'Timestamp')
            tree = ttk.Treeview(view_window, columns=columns, show='headings')
            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, width=100)
            tree.pack(fill='both', expand=True, padx=10, pady=10)

            for item in items:
                tree.insert('', 'end', values=(
                    item['prediction_id'],
                    item.get('name', 'N/A'),  # Include name
                    item['height_cm'],
                    item['weight_kg'],
                    f"{item['predicted_body_fat']:.2f}",
                    item['category'],
                    item['timestamp']
                ))

            def view_images():
                selected_item = tree.selection()
                if not selected_item:
                    messagebox.showwarning("Selection Error", "Please select a prediction to view images.")
                    return

                item_id = tree.item(selected_item)['values'][0]
                response = self.table.get_item(Key={'prediction_id': item_id})
                item = response.get('Item')
                if not item:
                    messagebox.showerror("Error", "Prediction data not found.")
                    return

                front_key = item['front_image_key']
                side_key = item['side_image_key']
                try:
                    front_obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=front_key)
                    side_obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=side_key)
                    front_image = Image.open(BytesIO(front_obj['Body'].read()))
                    side_image = Image.open(BytesIO(side_obj['Body'].read()))

                    image_window = tk.Toplevel(view_window)
                    image_window.title("Prediction Images")
                    front_tk = ImageTk.PhotoImage(front_image.resize((320, 240)))
                    side_tk = ImageTk.PhotoImage(side_image.resize((320, 240)))
                    tk.Label(image_window, image=front_tk).pack(side="left", padx=10)
                    tk.Label(image_window, image=side_tk).pack(side="right", padx=10)
                    image_window.image_front = front_tk
                    image_window.image_side = side_tk
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load images: {str(e)}")
                    print(f"Image load error: {e}")

            tk.Button(view_window, text="View Selected Images", command=view_images).pack(pady=10)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load past predictions: {str(e)}")
            print(f"Load predictions error: {e}")

    def close_camera(self):
        try:
            if self.camera is not None:
                self.camera.close()
                self.camera = None
                print("Camera closed successfully")
        except Exception as e:
            print(f"Camera close error: {str(e)}")

    def close_app(self):
        self.close_camera()
        for path in self.image_paths.values():
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"Error removing temp file {path}: {e}")
        self.root.destroy()

    def __del__(self):
        self.close_camera()

if __name__ == "__main__":
    root = tk.Tk()
    app = BodyFatApp(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.close_app()
