from re import M


def kmp_search(s, p):
    """
    KMP solution for searching pattern in a string
    Time Complexity: O(m+n)
    Space Complexity: O(m)
    args:
        s: target string
        p: pattern
    returns:
        is_match: a boolean value to show if the pattern exists in s
    """
    n = len(s)
    m = len(p)
    failure_function = [0] * m

    # Build failure function recording longest proper suffix and prefix
    # dp
    for i in range(1, m):
        j = failure_function[i-1]

        while p[j] != p[i]:
            if j == 0:
                break
            j = failure_function[j-1]

        if p[j] == p[i]:
            failure_function[i] = j + 1
        else:
            failure_function[i] = 0

    # kmp
    i = 0
    j = 0
    while i < n:
        if s[i] != p[j]:
            if j == 0:
                i += 1
                continue
            # Lookup the longest proper suffix and prefix before current character
            j = failure_function[j-1]
        else:
            i += 1
            j += 1

        if j == m:
            return True
    return False


#0123456
#abcdabc
#0000123

def failure_function(p):
    m = len(p)
    f = [0] * m
    # string 0~m pre-sufix longest count
    for i in range(1, m):
        j = f[i - 1]
        while p[i] != p[j]:
            if j == 0:
                break
            j = f[j - 1]

        if p[i] == p[j]:
            f[i] = j + 1
        else:
            f[i] = 0


failure_function("abcabb")