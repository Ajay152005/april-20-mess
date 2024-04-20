def max_area(height):
    max_area = 0
    left, right = 0, len(height) - 1

    while left < right:
        width = right - left
        container_height = min(height[left], height[right])
        area = width * container_height
        max_area = max(max_area, area)

        if height[left] < height[right]:
            left += 1
        else: 
            right -= 1
    return max_area

# Example usage:
height1 = [1,8,6,2,5,4,8,3,7]
print(max_area(height1))

height2 = [1,1]

print(max_area(height2))