import cv2
import numpy as np
import matplotlib.pyplot as plt


# using imread() original image
img = cv2.imread('image_capturee.jpg')
print(img.shape)

#BLUE channel
matriceB=np.zeros(img.shape, dtype=np.uint8)
matriceB[:,:,0]=img[:,:,0]
cv2.imwrite(r'C:\Users\MED Fedi BOUABID\Desktop\CR IIA3\TI et vision\canal_bleu.png',matriceB)

#GREEN channel
matriceG=np.zeros(img.shape, dtype=np.uint8)
matriceG[:,:,1]=img[:,:,1]
cv2.imwrite(r'C:\Users\MED Fedi BOUABID\Desktop\CR IIA3\TI et vision\canal_vert.png',matriceG)

#RED channel
matriceR=np.zeros(img.shape, dtype=np.uint8)
matriceR[:,:,2]=img[:,:,2]
cv2.imwrite(r'C:\Users\MED Fedi BOUABID\Desktop\CR IIA3\TI et vision\canal_rouge.png',matriceR)


blue_channel, green_channel, red_channel = cv2.split(img)

# Display using subplots
plt.figure(figsize=(10, 6))

# Original Image
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

# Blue Channel
plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(matriceB, cv2.COLOR_BGR2RGB))
plt.title('Blue Channel')

# Green Channel
plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(matriceG, cv2.COLOR_BGR2RGB))
plt.title('Green Channel')

# Red Channel
plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(matriceR, cv2.COLOR_BGR2RGB))
plt.title('Red Channel')

plt.tight_layout()
plt.show()
# Plot histograms for each channel
plt.figure(figsize=(10, 6))

plt.subplot(1,3, 1)
plt.hist(blue_channel.flatten(), bins=256, color='blue', alpha=0.7, rwidth=0.8)
plt.title('Blue Channel Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.subplot(1,3, 2)
plt.hist(green_channel.flatten(), bins=256, color='green', alpha=0.7, rwidth=0.8)
plt.title('Green Channel Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.subplot(1,3, 3)
plt.hist(red_channel.flatten(), bins=256, color='red', alpha=0.7, rwidth=0.8)
plt.title('Red Channel Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

cv2.waitKey(0);
cv2.destroyAllWindows();
cv2.waitKey(1)

cv2.imwrite(r'C:\Users\MED Fedi BOUABID\Desktop\CR IIA3\TI et vision\gray_green.png',matriceG)


############################### QUESTION 2 & 3 #######################################



# using imread() IMAGE EN QUESTION
img = cv2.imread('image_capturee.jpg')

#splitting the image into red blue green channels
imgB=img[:,:,0]
imgG=img[:,:,1]
imgR=img[:,:,2]

# Convert and save by taking the average of RGB values
avg_gray_img = np.mean(img, axis=2).astype(np.uint8)
cv2.imwrite(r'C:\Users\MED Fedi BOUABID\Desktop\CR IIA3\TI et vision\avg_gray_img.png',avg_gray_img)

# Convert and save using luminosity formula
lum_gray_img = (0.299 * img[:,:,2] + 0.587 * img[:,:,1] + 0.114 * img[:,:,0]).astype(np.uint8)
cv2.imwrite(r'C:\Users\MED Fedi BOUABID\Desktop\CR IIA3\TI et vision\lum_gray_img.png',lum_gray_img)

#convert and save using rgb2gray de opencv
imagegray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite(r'C:\Users\MED Fedi BOUABID\Desktop\CR IIA3\TI et vision\rgb2gray.png',imagegray)



# Display the original and grayscale images
plt.figure(figsize=(10, 5))

plt.subplot(2,2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(2,2, 2)
plt.imshow(avg_gray_img, cmap='gray')
plt.title('Grayscale Image (Average Method)')

plt.subplot(2,2, 3)
plt.imshow(lum_gray_img, cmap='gray')
plt.title('luminance Formula method Image')

plt.subplot(2,2, 4)
plt.imshow(imagegray, cmap='gray')
plt.title('Grayscale Image (rgb2gray)')

plt.tight_layout()
plt.show()

# Apply the specified transformations
binary_img = np.zeros_like(lum_gray_img)
binary_img[(lum_gray_img < 45 )]= 0
binary_img[(lum_gray_img >= 45) & (lum_gray_img < 66)] = 66
binary_img[(lum_gray_img >= 66) & (lum_gray_img < 110)] = 110
binary_img[(lum_gray_img >= 110) & (lum_gray_img < 165)] = 165
binary_img[(lum_gray_img >= 165)& (lum_gray_img < 200)] = 200
binary_img[(lum_gray_img >= 200)]=255

# Display the original and transformed images
plt.figure(figsize=(10, 5))

plt.subplot(2, 2, 1)
plt.imshow(lum_gray_img, cmap='gray')
plt.title('Original Image Luminance formula')

plt.subplot(2, 2, 2)
plt.imshow(binary_img, cmap='gray')
plt.title('seuil <45,<66,<110,<165,<200,<255')

plt.subplot(2, 2, 3)
plt.hist(lum_gray_img.flatten(), bins=256, color='blue', alpha=0.7, rwidth=0.8)
plt.title('luminance formula Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.subplot(2, 2, 4)
plt.hist(binary_img.flatten(), bins=256, color='blue', alpha=0.7, rwidth=0.8)
plt.title('seuil image Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
cv2.imwrite(r'C:\Users\MED Fedi BOUABID\Desktop\CR IIA3\TI et vision\binarized_image.png',binary_img)



############################# QUESTION 4-a ###################################

############################## REMARQUE ###########################################


# ICI ON FAIT LA MANIPULATION SUR L'IMAGE BINAIRE ,
# ON CHANGE ENSUITE LA LIGNE CI-DESSOUS binarized_image PAR lum_gray_img
# MEME REMARQUE POUR QUESTION 4-b


#import the gray scale image
lum_gray_img=cv2.imread('binarized_image.png',cv2.IMREAD_GRAYSCALE)

#detection de contour avec un filtre spatial

# Appliquer le noyau de Sobel and save the images
sobel_x = cv2.Sobel(lum_gray_img, cv2.CV_64F, 1, 0, ksize=3)
cv2.imwrite(r'C:\Users\MED Fedi BOUABID\Desktop\CR IIA3\TI et vision\sobel_x_binaire.png',sobel_x)
sobel_y = cv2.Sobel(lum_gray_img, cv2.CV_64F, 0, 1, ksize=3)
cv2.imwrite(r'C:\Users\MED Fedi BOUABID\Desktop\CR IIA3\TI et vision\sobel_y_binaire.png',sobel_y)

# Calculer le gradient total (approximation)
gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

# Binariser et sauvegarder l'image résultante pour obtenir les contours
threshold = 25  # Vous pouvez ajuster ce seuil selon vos besoins
contour_image = np.zeros_like(lum_gray_img)
contour_image[gradient_magnitude > threshold] = 255
cv2.imwrite(r'C:\Users\MED Fedi BOUABID\Desktop\CR IIA3\TI et vision\sobel_finale_binaire.png',contour_image)

# Afficher l'image originale et l'image avec les contours

plt.figure(figsize=(10, 5))
plt.subplot(2, 2, 1), plt.imshow(lum_gray_img, cmap='gray')
plt.title('Image originale'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 4), plt.imshow(contour_image, cmap='gray')
plt.title('Contours détectés'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 3), plt.imshow(sobel_x, cmap='gray')
plt.title('sobel x'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 2), plt.imshow(sobel_y, cmap='gray')
plt.title('sobel y'), plt.xticks([]), plt.yticks([])

plt.show()




############################## QUESTION 4-b ################################"





#import the gray scale image
lum_gray_img=cv2.imread('binarized_image.png',cv2.IMREAD_GRAYSCALE)

#detection de contour avec un filtre morphologique

# Définir un élément structurant de forme rectangle et de taille 3
element_structurant_rectangle = np.ones((5,5), dtype=np.uint8)

# Appliquer une opération de dilatation
dilated = cv2.dilate(lum_gray_img, element_structurant_rectangle, iterations=1)
cv2.imwrite(r'C:\Users\MED Fedi BOUABID\Desktop\CR IIA3\TI et vision\dilated_binaire.png',dilated)

# Appliquer une opération d'érosion
eroded = cv2.erode(lum_gray_img, element_structurant_rectangle, iterations=1)
cv2.imwrite(r'C:\Users\MED Fedi BOUABID\Desktop\CR IIA3\TI et vision\eroded_binaire.png',eroded)

# Calculer le gradient externe
gradient_external = cv2.subtract(dilated, lum_gray_img)
cv2.imwrite(r'C:\Users\MED Fedi BOUABID\Desktop\CR IIA3\TI et vision\gradient_external_binaire.png',gradient_external)

# Calculer le gradient interne
gradient_internal = cv2.subtract(lum_gray_img, eroded)
cv2.imwrite(r'C:\Users\MED Fedi BOUABID\Desktop\CR IIA3\TI et vision\gradient_internal_binaire.png',gradient_internal)

# Calculer le gradient complet
gradient_complete = cv2.subtract(dilated, eroded)
cv2.imwrite(r'C:\Users\MED Fedi BOUABID\Desktop\CR IIA3\TI et vision\gradient_complete_binaire.png',gradient_complete)

# Afficher les résultats
plt.figure(figsize=(15, 8))
plt.subplot(231), plt.imshow(lum_gray_img, cmap='gray'), plt.title('Image Originale')
plt.subplot(232), plt.imshow(dilated, cmap='gray'), plt.title('Image Dilatée')
plt.subplot(233), plt.imshow(eroded, cmap='gray'), plt.title('Image Érodée')
plt.subplot(234), plt.imshow(gradient_external, cmap='gray'), plt.title('Gradient Externe')
plt.subplot(235), plt.imshow(gradient_internal, cmap='gray'), plt.title('Gradient Interne')
plt.subplot(236), plt.imshow(gradient_complete, cmap='gray'), plt.title('Gradient Complet')
plt.show()
