import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("VIRTUoss")
        self.root.geometry("1000x700")

        self.ref_image_path = None
        self.target_image_path = None
        self.ref_points = []
        self.target_points = []

        # Home Page
        self.show_home_page()

    def show_home_page(self):
        # Clear the window
        for widget in self.root.winfo_children():
            widget.destroy()

        description = tk.Label(self.root, text="Welcome !\n\nThis app allows you to register Images and Videos based on a 4 reference points.", font=("Arial", 14))
        description.pack(pady=20)

        get_started_btn = tk.Button(self.root, text="Start", font=("Arial", 12), command=self.show_image_selection_page)
        get_started_btn.pack(pady=10)

    def show_image_selection_page(self):
        # Clear the window
        for widget in self.root.winfo_children():
            widget.destroy()

        # container for the two sections
        container = tk.Frame(self.root)
        container.pack(fill="both", expand=True, padx=20, pady=10)

        # Left Column (Reference Image)
        self.left_column = tk.Frame(container)
        self.left_column.pack(side="left", fill="both", expand=True, padx=10)

        # Reference Image Section
        self.ref_frame = tk.LabelFrame(self.left_column, text="Reference Image", padx=10, pady=10)
        self.ref_frame.pack(fill="both", expand=True, pady=10)

        self.ref_image_label = tk.Label(self.ref_frame, text="No image selected", font=("Arial", 12))
        self.ref_image_label.pack()

        self.ref_select_btn = tk.Button(self.ref_frame, text="Select Reference Image", command=lambda: self.select_image("reference"))
        self.ref_select_btn.pack(pady=5)

        # Right Column (Target Image)
        self.right_column = tk.Frame(container)
        self.right_column.pack(side="right", fill="both", expand=True, padx=10)

        # Target Image Section
        self.target_frame = tk.LabelFrame(self.right_column, text="Target Image", padx=10, pady=10)
        self.target_frame.pack(fill="both", expand=True, pady=10)

        self.target_image_label = tk.Label(self.target_frame, text="No image selected", font=("Arial", 12))
        self.target_image_label.pack()

        self.target_select_btn = tk.Button(self.target_frame, text="Select Target Image", command=lambda: self.select_image("target"))
        self.target_select_btn.pack(pady=5)

        # Transformation Button
        transform_btn = tk.Button(self.root, text="Registration", font=("Arial", 12), command=self.launch_transformation)
        transform_btn.pack(pady=20)

    def select_image(self, image_type):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            image = Image.open(file_path)
            image.thumbnail((300, 300))
            photo = ImageTk.PhotoImage(image)

            if image_type == "reference":
                self.ref_image_path = file_path
                self.ref_image_label.config(image=photo)
                self.ref_image_label.image = photo
                self.ref_image_label.text = ""
                self.show_point_selection("reference")
            elif image_type == "target":
                self.target_image_path = file_path
                self.target_image_label.config(image=photo)
                self.target_image_label.image = photo
                self.target_image_label.text = ""
                self.show_point_selection("target")

    def show_point_selection(self, image_type):
        # Clear previous point selection widgets
        if image_type == "reference":
            for widget in self.ref_frame.winfo_children():
                if widget not in [self.ref_image_label, self.ref_select_btn]:
                    widget.destroy()
        elif image_type == "target":
            for widget in self.target_frame.winfo_children():
                if widget not in [self.target_image_label, self.target_select_btn]:
                    widget.destroy()

        # Point Type Selection
        point_type_frame = tk.Frame(self.ref_frame if image_type == "reference" else self.target_frame)
        point_type_frame.pack(pady=10)

        self.point_type = tk.StringVar(value="manual")  # Default to manual input

        manual_radio = tk.Radiobutton(point_type_frame, text="Manually Enter Points", variable=self.point_type, value="manual", command=lambda: self.toggle_point_input(image_type))
        manual_radio.pack(anchor="w")

        select_radio = tk.Radiobutton(point_type_frame, text="Select Points from Image", variable=self.point_type, value="select", command=lambda: self.toggle_point_input(image_type))
        select_radio.pack(anchor="w")

        # Manual Input Fields 
        self.manual_input_frame = tk.Frame(self.ref_frame if image_type == "reference" else self.target_frame)
        self.manual_input_frame.pack(fill="x", pady=10)

        self.input_fields = self.create_input_fields(self.manual_input_frame, image_type)
        self.toggle_point_input(image_type)

    def create_input_fields(self, parent, image_type):
        frame = tk.LabelFrame(parent, text=f"{image_type.capitalize()} Points", padx=10, pady=10)
        frame.pack(fill="x", pady=5)

        labels = ["Point 1 (x, y):", "Point 2 (x, y):", "Point 3 (x, y):", "Point 4 (x, y):"]
        entries = []
        for label in labels:
            row = tk.Frame(frame)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=label, width=15).pack(side="left")
            entry = tk.Entry(row)
            entry.pack(side="left", expand=True, fill="x")
            entry.bind("<KeyRelease>", lambda event, img_type=image_type: self.update_points_on_image(img_type))
            entries.append(entry)
        return entries

    def toggle_point_input(self, image_type):
        if self.point_type.get() == "manual":
            self.manual_input_frame.pack()
            if image_type == "reference":
                self.ref_select_points_btn.pack_forget() if hasattr(self, "ref_select_points_btn") else None
            else:
                self.target_select_points_btn.pack_forget() if hasattr(self, "target_select_points_btn") else None
        else:
            self.manual_input_frame.pack_forget()
            if image_type == "reference":
                self.ref_select_points_btn = tk.Button(self.ref_frame, text="Select Reference Points", command=lambda: self.select_points("reference"))
                self.ref_select_points_btn.pack(pady=5)
            else:
                self.target_select_points_btn = tk.Button(self.target_frame, text="Select Target Points", command=lambda: self.select_points("target"))
                self.target_select_points_btn.pack(pady=5)

    def update_points_on_image(self, image_type):
        if image_type == "reference":
            points = [tuple(map(float, entry.get().split(','))) for entry in self.input_fields if entry.get()]
            self.ref_points = points
            self.display_points_on_image("reference", points)
        else:
            points = [tuple(map(float, entry.get().split(','))) for entry in self.input_fields if entry.get()]
            self.target_points = points
            self.display_points_on_image("target", points)

    def display_points_on_image(self, image_type, points):
        image_path = self.ref_image_path if image_type == "reference" else self.target_image_path
        if not image_path:
            return

        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        for point in points:
            draw.ellipse((point[0]-5, point[1]-5, point[0]+5, point[1]+5), fill="red")
        image.thumbnail((300, 300))
        photo = ImageTk.PhotoImage(image)

        if image_type == "reference":
            self.ref_image_label.config(image=photo)
            self.ref_image_label.image = photo
        else:
            self.target_image_label.config(image=photo)
            self.target_image_label.image = photo

    def select_points(self, image_type):
        if image_type == "reference" and not self.ref_image_path:
            messagebox.showerror("Error", "Please select a reference image first.")
            return
        if image_type == "target" and not self.target_image_path:
            messagebox.showerror("Error", "Please select a target image first.")
            return

        # Open a new window to select points
        points_window = tk.Toplevel(self.root)
        points_window.title(f"Select {image_type.capitalize()} Points")
        points_window.geometry("800x600")

        image_path = self.ref_image_path if image_type == "reference" else self.target_image_path
        image = Image.open(image_path)
        image.thumbnail((800, 600))
        photo = ImageTk.PhotoImage(image)

        canvas = tk.Canvas(points_window, width=800, height=600)
        canvas.pack()
        canvas.create_image(0, 0, anchor="nw", image=photo)
        canvas.image = photo

        points = []

        def on_click(event):
            x, y = event.x, event.y
            points.append((x, y))
            canvas.create_oval(x-5, y-5, x+5, y+5, fill="red")
            if len(points) == 4:
                if image_type == "reference":
                    self.ref_points = points
                else:
                    self.target_points = points
                messagebox.showinfo("Points Selected", f"4 points selected for {image_type}.")
                points_window.destroy()
                self.display_points_on_image(image_type, points)

        canvas.bind("<Button-1>", on_click)

    def launch_transformation(self):
        messagebox.showinfo("Transformation", f"Transformation launched!\nReference Points: {self.ref_points}\nTarget Points: {self.target_points}")

# Run
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()