from tkinter import Tk, Label, Button, filedialog, messagebox
from PIL import Image, ImageTk
from convert_jpgs_to_binary import convert_one
from eval_single import evaluate
class MyFirstGUI:
    image_path = None
    photo = None
    def __init__(self, master):
        self.master = master
        self.photo = None
        master.title("Fruit recognition")
        self.image_label = Label(self.master)
        self.image_label.pack()

        self.label = Label(master, text="unknown")
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

        name = evaluate()

        self.label['text'] = "\n".join([" -> ".join(x) for x in name])

    def load_file(self):
        filename = filedialog.askopenfilename(filetypes =( ("All files", "*.*")
                                                             ,("BMP files", "*.bmp")
                                                             ,("PNG files", "*.png")
                                                             ,("JPEG files", "*.jpeg;.jpg")))
        if filename:
            try:
                self.image_path = filename
                image = Image.open(self.image_path)
                image.thumbnail((100,100), Image.ANTIALIAS)

                if image.size[0] != image.size[1]:
                    raise Exception("only square images work at this time")
                self.photo = ImageTk.PhotoImage(image)
                self.image_label.config(image=self.photo)
            except Exception as e:
                messagebox.showerror("Open Image File", "Failed to read file \n'%s', %s" % (filename,str(e)))
                return
root = Tk()
my_gui = MyFirstGUI(root)
root.mainloop()