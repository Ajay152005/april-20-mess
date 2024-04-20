def longest_common_prefix(strs):
    if not strs:
        return ''
    
    # Assume the first string as the prefix

    prefix = strs[0]

    for i in range(1, len(strs)):
        # Compare each character of the prefix with the corresponding character of the current string
        j = 0
        while j < len(prefix) and j < len(strs[i]) and prefix[j] == strs[i][j]:
            j += 1
        #update the prefix to the common prefix up to the current string

        prefix = prefix[:j]

        # if the prefix becomes empty, there's no common prefix, so we can break early
        if not prefix:
            break
    return prefix

# EXAMPLE usage:
strs1 = ['flower', 'flow', 'flight']
print(longest_common_prefix(strs1)) #output: "f1"

strs2 = ["dog", "racecar","car"]
print(longest_common_prefix(strs2)) #output: ""