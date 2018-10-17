class Solution:
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """

        res = list(map(lambda x: int(x), list(str(int(''.join(map(lambda x:str(x), digits))) + 1))))
        return res

test = Solution().plusOne
assert test([1, 2, 3]) == [1, 2, 4]
print('All passed!')
