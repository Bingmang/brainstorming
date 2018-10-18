class Solution:
    def reverseString(self, s):
        """
        :type s: str
        :rtype: str
        """
        return s[::-1]
        
test = Solution().reverseString
assert test('hello') == 'olleh'
print('All passed!')
