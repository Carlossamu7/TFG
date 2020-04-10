from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def display_image(img):
# Display a single image
    plt.gray()
    plt.imshow(img)
    plt.show()

def show_images(images):
# Display all the images
    titles = ["Original", "Rows Only", "2D Haar Transform", "Return to Original"]
    fig = plt.figure(figsize=(9, 9)) #(12, 3)
    for i, a in enumerate(images):
        display = fig.add_subplot(2, 2, i + 1) #(1, 4, i + 1)
        display.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        display.set_title(titles[i], fontsize=10)
        display.set_xticks([])
        display.set_yticks([])

    fig.tight_layout()
    plt.show()


def list_split(list, length, offset):
# splits the list in order to do haar transform, recursive function
# returns the haar transformed list of pixels
    if( length > 2 ):
        first = list[:len(list)//2]
        second = list[len(list)//2:]

        offset = second + offset
        n = length / 2
        first = haar_transform(first)
        return list_split(first, n, offset)
    else:
        return list + offset


def haar_transform(list):
# performs the haar algorithm on a list
# then returns the transformed list
    avgs = []
    diffs = []
    i = 0
    for pixel in list:
        if (i % 2 == 0):
            avg = ((list[i] + list[i+1]) / 2 ) #np.sqrt(2))
            avgs.append(avg)

            diff = ((list[i] - list[i+1]) / 2 ) #np.sqrt(2))
            diffs.append(diff)
        i += 1
    haar = avgs + diffs
    return haar

############################ Getting original image back #######################

def initial_values(right, i):
# we have to do this once for every list
# does the reverse_haar but on the first 2 values of the list
    list = []
    first = right[i] + right[i+1]
    list.append(first)
    second = right[i] - right[i+1]
    list.append(second)
    return list

def reverse_haar(front, the_rest, i, power):
# returns the original list of pixels, reverses the algorithm
    reverse = []
    while( i < len(front)):
        sum = front[i] + the_rest[i]
        reverse.append(sum)
        difference = front[i] - the_rest[i]
        reverse.append(difference)
        i += 1

    if(len(the_rest) > len(reverse) ):
        the_rest = the_rest[2**power:]
        power += 1
        i = 0
        return reverse_haar(reverse, the_rest, i, power)
    else:
        return reverse

################################################################################

image_chosen = input("Image name: ")
image_chosen = "images/" + image_chosen
resolution = int(input("Dimension of image (256, 512, 1024, 2048, etc.): "))

images_list = []
# opens an image
im = Image.open(image_chosen, 'r')
# gets the pixels, stores them in pix_val
pix_val = list(im.getdata())
# turns the list into a string object
line = ','.join(str(v) for v in pix_val)
string = line.split(',');
# nums holds the a list of ints
nums = [int(i) for i in string]
# Prints original image
og = np.array(pix_val)
img = og.reshape(resolution, resolution)
print("\n")
print(img)
# display_image(img)
images_list.append(img)
# this gives me a list of length of dimension given, each index position holds a list of (256, 512, 1024 ...)
split = list(zip(*[iter(nums)] * resolution))
print("\nNumber of entries in split: " , len(split))

print("length of items in list: " + str(len(split[0])))
print("\n")

offset = []
combined = []
print("----> First transform on Rows Only\n:")

for list in split:
    haar = haar_transform(list)
    combined.append(list_split(haar, len(haar), offset))

rows = np.array(combined)
img_rows = rows.reshape(resolution, resolution)
print(img_rows)
# display_image(img_rows)
images_list.append(img_rows)
print("\n")
print("The columns of the first transform are done next")

columns_rows = []
# Transpose of img_rows
transpose_columns = [*zip(*img_rows)]
for column in transpose_columns:
    haar2 = haar_transform(column)
    columns_rows.append(list_split(haar2, len(haar2), offset))

columns = np.array(columns_rows)
haar_image = columns.reshape(resolution, resolution)

print(haar_image)
# display_image(haar_image)
images_list.append(haar_image)

################### transpose the array #########################
i = 0
rev_columns = []
power = 1
for pixels in haar_image:
    right = pixels[:2]
    the_rest = pixels[2::]
    init_val = initial_values(right, i)
    rev_columns.append(reverse_haar(init_val, the_rest, i, power))

print("\n>>>> Rows Again <<<<\n")
rev_rows = [*zip(*rev_columns)]
rev_haar_image = []
i = 0
power = 1
for pixels in rev_rows:
    right = pixels[:2]
    the_rest = pixels[2::]
    init_val = initial_values(right, i)
    rev_haar_image.append(reverse_haar(init_val, the_rest, i, power))

rev_image = np.array(rev_haar_image)
rev_og = rev_image.reshape(resolution, resolution)
print(rev_og)
# display_image(rev_og)
images_list.append(rev_og)
show_images(images_list)
