import streamlit as st

# Text/Title
st.title("Streamlit tutorials")

# Heeader/Subheaders
st.header("This is a header")
st.subheader("This is a subheader")

# Text
st.text("Hello Streamlit")

# Markdown

st.markdown("### This is a markdown")

# Error/Colorful Text
st.success("Successful")

st.info("Information!")

st.warning("This is a warning")

st.error("This is an error Danger")

st.exception("NameError('name three not defined')")

# Get help info about Python
st.help(range)

# Writing Text
st.write("Text with write")
st.write(range(10))


# # Images
# from PIL import Image
# img = Image.open("example.jpeg")
# st.image(img, width=300, caption="Simple Image")

# # Videos
# vid_file = open("example.mp4", "rb").read()
# # vic_byte = vid_rile.read()
# st.video(vid_file)

# # Audio
# audio_file = open("Examplemusic.mp3", "rb").read()
# st.audio(audio_file, format='audio/mp3')


# Widget

# Check box
if st.checkbox("Show/Hide"):
  st.text("Showing or Hiding Widget")
  
# Radio
status = st.radio("What is your status", ("Active", "Inactive"))

if status == "Active":
  st.success("You're active")
else:
  st.warning("Inactive, Activate")

# SelectBox
occupation = st.selectbox("Your Occupation", ["Programmer", "Datascientist", "Doctor", "Businessman"])
st.write("You selected this option ", occupation)

# MultiSelect
location = st.multiselect("Where do you work?", ["London", "New York", "San Francisco", "Nepal"])
st.write("You selected", len(location), "locations.")

# Slider
age = st.slider("What is your age?", 1, 90)

# Buttons
st.button("Simple Button")

if st.button("About"):
  st.text("Streamlit is cool")

# Text Input
name = st.text_input("Enter your name", "Type here...")
if st.button("Submit"):
  result = name.title()
  st.success(result)