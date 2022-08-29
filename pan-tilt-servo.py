# Blob Detection and uart transport
import sensor, image, time
from pyb import UART
import json
# For color tracking to work really well you should ideally be in a very, very,
# very, controlled enviroment where the lighting is constant...
green_threshold   = (28, 14, -35, -16, -57, 49)
# You may need to tweak the above settings for tracking green things...
# Select an area in the Framebuffer to copy the color settings.

sensor.reset() # Initialize the camera sensor.
sensor.set_pixformat(sensor.RGB565) # use RGB565.
sensor.set_framesize(sensor.QQVGA) # use QQVGA for speed.
sensor.skip_frames(10) # Let new settings take affect.
sensor.set_auto_whitebal(False) # turn this off.
clock = time.clock() # Tracks FPS.

uart = UART(3, 115200)
def find_max(blobs):
    max_size=0
    for blob in blobs:
        if blob.pixels() > max_size:
            max_blob=blob
            max_size = blob.pixels()
    return max_blob

while(True):
    img = sensor.snapshot() # Take a picture and return the image.

    blobs = img.find_blobs([green_threshold])
    if blobs:
        max_blob=find_max(blobs)
        print('sum :', len(blobs))
        img.draw_rectangle(max_blob.rect())
        img.draw_cross(max_blob.cx(), max_blob.cy())

        output_str="{%d,%d,%d,%d,}" % (max_blob.cx(),img.width()/2, max_blob.cy(),img.height()/2)

        print('Sending:',output_str)
        uart.write(output_str)
    else:
        print('not found!')
