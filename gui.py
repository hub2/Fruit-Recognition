from tkinter import Tk, Label, Button, filedialog, messagebox
from PIL import Image, ImageTk
from convert_jpgs_to_binary import convert_one
from eval_single import evaluate
class MyFirstGUI:
    image_path = None
    photo = None
    def __init__(self, master):
        self.master = master
        master.title("A simple GUI")

        self.label = Label(master, text="This is our first GUI!")
        self.label.pack()

        self.load_file_button = Button(master, text="Load File", command=self.load_file)
        self.load_file_button.pack()

        self.load_file_button = Button(master, text="Evaluate", command=self.evaluate)
        self.load_file_button.pack()

        self.close_button = Button(master, text="Close", command=master.quit)
        self.close_button.pack()

    def evaluate(self):
        if not self.image_path:
            return
        convert_one(self.image_path, "tmp")
        print("Converted")
        print(evaluate())

    def load_file(self):
        filename = filedialog.askopenfilename(filetypes =( ("All files", "*.*")
                                                             ,("BMP files", "*.bmp")
                                                             ,("PNG files", "*.png")
                                                             ,("JPEG files", "*.jpeg;.jpg")))
        if filename:
            try:
                self.image_path = filename
                image = Image.open(self.image_path)
                self.photo = ImageTk.PhotoImage(image)
            except:
                messagebox.showerror("Open Image File", "Failed to read file \n'%s'" % filename)
                return
root = Tk()
my_gui = MyFirstGUI(root)
root.mainloop()