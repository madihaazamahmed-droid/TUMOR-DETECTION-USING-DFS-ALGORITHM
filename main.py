import cv2
import numpy as np
def track(msg):
    print("[INFO] " + msg)
def dfs(x, y, visited, binary, component):
    stack = [(x, y)]
    h, w = binary.shape

    while stack:
        cx, cy = stack.pop()

        if visited[cx][cy] == 1:
            continue

        visited[cx][cy] = 1
        component.append((cx, cy))
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < h and 0 <= ny < w:
                if binary[nx][ny] == 255 and visited[nx][ny] == 0:
                    stack.append((nx, ny))
def detect_tumor(image_path):

    track("Loading image...")
    img = cv2.imread(image_path)

    if img is None:
        track("Error: Image not found!")
        return
    
    track("Image loaded successfully.")
    track("Converting to grayscale...")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    track("Applying threshold...")
    _, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    track("Starting DFS to detect tumor region...")

    visited = np.zeros_like(binary)
    h, w = binary.shape

    largest_component = []
    max_size = 0

    for i in range(h):
        for j in range(w):
            if binary[i][j] == 255 and visited[i][j] == 0:
                component = []
                dfs(i, j, visited, binary, component)

                if len(component) > max_size:
                    max_size = len(component)
                    largest_component = component

    track("DFS completed.")
    track(f"Tumor region size: {max_size} pixels")
    mask = np.zeros_like(gray)
    for (x, y) in largest_component:
        mask[x][y] = 255
    track("Highlighting tumor region...")
    detected = img.copy()
    detected[mask == 255] = [0, 0, 255] 
    track("Displaying results... (Close windows to continue)")
    cv2.imshow("Original MRI", img)
    cv2.imshow("Binary Threshold", binary)
    cv2.imshow("Tumor Mask (DFS)", mask)
    cv2.imshow("Detected Tumor", detected)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    output_path = "tumor_output.jpg"
    cv2.imwrite(output_path, detected)
    track(f"Output saved as: {output_path}")
detect_tumor("brain.jpg")
