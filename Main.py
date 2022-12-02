import API

def start_up():
    # Create a new window
    API.create_window()
    root = API.root
    # Create the train4D frame and the train6D frame
    API.create_frames()
    train4D_frame = API.train4D_frame # Frame
    train6D_frame = API.train6D_frame # Frame
    # Create the Menu
    API.create_menu()

    # Set the content of tow frames
    API.set_frames()

    # Show the Frame
    train4D_frame.pack(fill='both', expand=1)
    # train6D_frame.pack(fill='both', expand=1)

    # Show the app window
    root.mainloop()

if __name__ == '__main__':
    start_up()