
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.*;

public class Main {

    public static void main(String[] args) {

        Main main = new Main();
//        int[] security = {1,2,5,4,1,0,2,4,5,3,1,2,4,3,2,4,8};
//        int time = 2;
//        System.out.println(main.goodDaysToRobBank_(security,time));
//        int[] nums = {0,1,0};
//        System.out.println(main.findMaxLength_(nums));
//        int[] w = {3,14,1,7};
//        int[] nums = {1,2,3,4};
//        //2 1 3 4---1 3 2 4---1 2 4 3---
//        //3 1 2 4---3 2 1 4---1 4 3 2---1 4 2 3---3 4 1 2---3 4 2 1
//        int[] nums2 = {1,2,3};
//        //
//        Main main1 = new Main(w);
//        while (true){
//            System.out.println(main1.pickIndex_());
//        }
//        int[] nums = {1,7,3,6,5,6};
//        int[] nums = {1,2,3};
//        int[] nums = {2,1,-1};
//        int[] ages = {20,30,100,110,120};
//        int[] nums = {1,0,1,0,1};
//        System.out.println(main.longestOnes_(nums,2));
//        String date = "2019-03-10";
//        System.out.println(main.dayOfYear_(date));
//        String s = "krrgw";
//        String t = "zjxss";
//        int maxCost = 19;
//        System.out.println(main.equalSubstring(s,t,maxCost));
//        int[] arr = {4,8,2,10};
//        int[][] queries = {{2,3},{1,3},{0,0},{0,3}};
//        System.out.println(Arrays.toString(main.xorQueries(arr, queries)));
//        int[] arr = {1,2};
//        int[][] matrix = {{5,2},{1,6}};
//        System.out.println(main.kthLargestValue(matrix, 4));
//        int[] nums = {2,-5,1,-4,3,-2};
//        System.out.println(main.maxAbsoluteSum(nums));
//        int[] nums = {9930,9923,9983,9997,9934,9952,9945,9914,9985,9982,9970,9932,9985,9902,9975,
//                9990,9922,9990,9994,9937,9996,9964,9943,9963,9911,9925,9935,9945,9933,9916,9930,
//                9938,10000,9916,9911,9959,9957,9907,9913,9916,9993,9930,9975,9924,9988,9923,9910,
//                9925,9977,9981,9927,9930,9927,9925,9923,9904,9928,9928,9986,9903,9985,9954,9938,
//                9911,9952,9974,9926,9920,9972,9983,9973,9917,9995,9973,9977,9947,9936,9975,9954,
//                9932,9964,9972,9935,9946,9966}; int k = 3056;
//        System.out.println(main.maxFrequency(nums,k));
//        System.out.println(1.0*0/0);
//        int[] nums = {1}; int k = 0;
//        System.out.println(main.subarraySum_prefix_hash(nums,k));
//        System.out.println(new Random().nextInt(10));
//        int[][] grid = {{1,2,5},{3,2,1}};
//        System.out.println(main.maxValue(grid));
//        String s = "AAAAAAAAAAA";
//        System.out.println(main.findRepeatedDnaSequences(s));
//        int num = 2;
//        System.out.println(Integer.toBinaryString(num));
//        System.out.println(Integer.toBinaryString((num << 2) | 2 & ((1 << 20) - 1)));
//        System.out.println(main.isPalindromicSubsequences("aba"));
//        String s = "abcdabcdabcdabcdabcdabcdabcdabcddcbadcbadcbadcbadcbadcbadcbadcba";
//        System.out.println(main.countPalindromicSubsequences(s));
//        System.out.println(Long.MAX_VALUE);
//        int[] asteroids = {5, 10, -5};
//        System.out.println(Arrays.toString(main.asteroidCollision(asteroids)));
//        String[] words = {"leetcoder","leetcode","od","hamlet","am"};
//        System.out.println(main.stringMatching(words));
//        System.out.println(Integer.toString(Integer.MAX_VALUE).length());
//        String str = new String(Integer.MAX_VALUE);
//        System.out.println(main.rotation("01234", 2));
//        char c = '9';
//        System.out.println((c+'4'));
//        int[] nums = {5,4,0,3,1,6,2};
//        System.out.println(main.arrayNesting(nums));
//        String sentence = "love errichto jonathan dumb";
//        String searchWord = "dumb";
//        System.out.println(main.isPrefixOfWord(sentence, searchWord));
//        ListNode listNode3 = new Main.ListNode(3);
//        ListNode listNode2 = new Main.ListNode(4,listNode3);
//        ListNode listNode1 = new Main.ListNode(0);
//        ListNode listNode6 = new Main.ListNode(4);
//        ListNode listNode5 = new Main.ListNode(3);
//        ListNode listNode4 = new Main.ListNode(7,listNode5);
//        System.out.println(main.addTwoNumbers(listNode1, listNode4));
//        int[] target = {3,7,9};
//        int[] arr = {3,7,11};
//        System.out.println(main.canBeEqual(target, arr));
//        String s = "a";
//        System.out.println(main.longestPalindrome(s));
//        int[] nums = {1,1,2,2,2,3};
//        System.out.println(Arrays.toString(main.frequencySort(nums)));
//        int[] nums = {4,3,2,3,5,2,1};
//        int k = 4;
//        System.out.println(main.canPartitionKSubsets(nums,k));
//        int[] nums = {1,2,3,4,5,6};
//        int key = 3;
//
//        System.out.println(main.binarySearch4Part(nums,key));
//        System.out.println(main.isPalindrome(121));

//        System.out.println(main.romanToInt("MMMCCCXXXIII"));
//        int[] nums = {2};
//        int target = 2;
//        System.out.println(Arrays.toString(main.searchRange_(nums, target)));
//        int x = 2147395599;
//        System.out.println(main.mySqrt_(x));
//        int num = 16;
//        System.out.println(main.isPerfectSquare(14));
//        int[] nums = {0,1,2,2,3,0,4,2};
//        int val = 2;
//        System.out.println(main.removeElement(nums,val));
//        int[] nums = {0,0,1};
//        System.out.println(main.removeDuplicates(nums));
//        String s = "y#fo##f";
//        String t = "y#f#o##f";
//        System.out.println(main.backspaceCompare(s,t));
//        int[] nums = {-1};
//        System.out.println(Arrays.toString(main.sortedSquares_(nums)));
//        int target = 7;
//        int[] nums = {2,3,1,2,4,3};
//        System.out.println(main.minSubArrayLen___(target,nums));
//        int[] fruits = {1,2,1};
//        System.out.println(main.totalFruit(fruits));
        String s = "a";
        String t = "a";
        System.out.println(main.minWindow(s,t));

    }

    public Main(){
    }


    public int[] twoSum(int[] nums, int target) {
        int[] result = new int[2];
        int len = nums.length;
        for (int i = 0; i < len; i++){
            for (int j = i+1; j < len; j++){
                if(nums[i]+nums[j] == target){
                    result[0] = i;
                    result[1] = j;
                    return result;
                }
            }
        }
        return null;
    }

    /**
     * 2100. 适合打劫银行的日子(超时)
     * @param security
     * @param time
     * @return
     */
    public List<Integer> goodDaysToRobBank(int[] security, int time) {
        List<Integer> results = new ArrayList<>();
        int len = security.length;
        for (int i = time; i < len-time;i++){
            boolean flag = false;
            for (int j = i - time; j < i; j++){
                if (security[j] < security[j+1]) {
                    flag = true;
                    break;
                }
            }
            for (int k = i; k < i+time; k++){
                if (security[k] > security[k+1]) {
                    flag = true;
                    break;
                }
            }
            if(!flag){
                results.add(i);
            }
        }
        return results;
    }

    /**
     * 2100. 适合打劫银行的日子(动态规划)
     * @param security
     * @param time
     * @return
     */
    public List<Integer> goodDaysToRobBank_df(int[] security,int time){
        int n = security.length;
        int[] left = new int[n];
        int[] right = new int[n];
        for (int i = 1; i < n; i++){
            if(security[i] <= security[i-1]){
                left[i] = left[i-1]+1;
            }
            if(security[n-i-1] <= security[n-i]){
                right[n-i-1] = right[n-i]+1;
            }
        }
        List<Integer> results = new ArrayList<>();
        for (int i = time; i < n-time; i++){
            if(left[i] >= time && right[i] >= time){
                results.add(i);
            }
        }
        return results;
    }

    /**
     *
     * 2100. 适合打劫银行的日子(前缀和)
     * @param security
     * @param time
     * @return
     */
    public List<Integer> goodDaysToRobBank_presum(int[] security,int time) {
        List<Integer> results = new ArrayList<>();
        int n = security.length;
        int[] non = new int[n];
        for (int i = 1; i < n; i++) {
            if (security[i - 1] == security[i]) continue;
            non[i] = security[i - 1] < security[i] ? 1 : -1;//1代表非递减，-1代表非递增
        }
        int[] nonincre = new int[n + 1];//非递增个数
        int[] nondecre = new int[n + 1];//非递减个数
        for (int i = 1; i <= n; i++) {
            nonincre[i] = nonincre[i - 1] + (non[i - 1] == 1 ? 1 : 0);
            nondecre[i] = nondecre[i - 1] + (non[i - 1] == -1 ? 1 : 0);
        }
        for (int i = time; i < n - time; i++) {
            int c1 = nonincre[i + 1] - nonincre[i + 1 - time];
            int c2 = nondecre[i + 1 + time] - nondecre[i + 1];
            if (c1 == 0 && c2 == 0) {
                results.add(i);
            }
        }
        return results;
    }

    /**
     * 525. 连续数组(超时)
     * @param nums
     * @return
     */
    public int findMaxLength(int[] nums) {

        int n = nums.length;
        int[] prefixNums = new int[n+1];
        for (int i = 1; i <= n; i++){
            prefixNums[i] += prefixNums[i-1]+nums[i-1];
        }
        int maxCount = 0;
        for (int i = n; i >= 1; i--){
            for (int j = 0; j < i; j++){
                if((i-j)%2==0){
                    if((i-j)/2 == prefixNums[i]-prefixNums[j]){
                        maxCount = Math.max(maxCount,i-j);
                    }
                }
            }
        }
        return maxCount;
    }


    /**
     * 525. 连续数组(前缀和+哈希)
     * @return
     */
    public int findMaxLength_(int[] nums) {
        HashMap<Integer,Integer> map = new HashMap<>();
        int n = nums.length;
        int current = 0;
        int maxCount = 0;
        map.put(0,-1);
        for (int i = 0; i < n; i++){
            if(nums[i] == 0){
                current-=1;
            }else {
                current+=1;
            }
            if(map.containsKey(current)){
                maxCount = Math.max(i - map.get(current),maxCount);
            }else{
                map.put(current,i);
            }
        }
        return maxCount;
    }


    /**
     * 528. 按权重随机选择
     */
    int[] prefix = null;
    int total = 0;
    int n = 0;
    public Main(int[] w) {
        n = w.length;
        prefix = new int[n+1];
        for (int i = 1; i <= n; i++){
            prefix[i] = prefix[i-1]+w[i-1];
        }
        total = Arrays.stream(w).sum();
    }

    public int pickIndex() {
        int num = (int) (Math.random()*total)+1;
        int result = 0;
        for (int i = 1; i <= n; i++) {
            if (num >= prefix[i - 1] && num < prefix[i]) {
                result = i - 1;
                break;
            }
        }
        return result;
    }

    public int pickIndex_(){
        int num = (int)(Math.random()*total)+1;
        return binarySearch(num);
    }

    public int binarySearch(int x){
        int low = 0; int high = n-1;
        while (low < high){
            int mid = (low+high)/2;
            if(prefix[mid] < x){
                low = mid+1;
            }else{
                high = mid-1;
            }
        }
        return low;
    }


    /**
     * 724. 寻找数组的中心下标
     * @param nums
     * @return
     */
    public int pivotIndex(int[] nums) {
        int pivot = -1;
        int n = nums.length;
        int[] prefix = new int[n+1];
        for (int i = 1; i <= n;i++){
            prefix[i] = prefix[i-1]+nums[i-1];
        }
        for (int i = 1; i <= n;i++){
            if(prefix[i-1] == prefix[n]-prefix[i]){
                pivot=i-1;
                break;
            }
        }
        return pivot;
    }


    /**
     * 825. 适龄的朋友(暴力)
     * @param ages
     * @return
     */
    public int numFriendRequests(int[] ages) {
        int n = ages.length;
        int count = 0;
        for (int i = 0; i < n; i++){
            for (int j = 0; j < n; j++){
                if(i != j){
                    if ((ages[i] <= 0.5*ages[j] + 7) || (ages[i] > ages[j]) || (ages[i] > 100 && ages[j] < 100)){

                    }else {
                        count++;
                    }
                }
            }
        }
        return count;
    }


    /**
     * 825. 适龄的朋友(排序+双指针)
     * @param ages
     * @return
     */
    public int numFriendRequests_(int[] ages) {
        int n = ages.length;
        int count = 0; int left = 0; int right = 0;
        Arrays.sort(ages);
        for (int age:ages){
            if(age < 15){
                continue;
            }
            while (ages[left] <= 0.5*age+7){
                ++left;
            }
            while (right+1 < n && ages[right+1] <= age){
                ++right;
            }
            count+=right-left;
        }
        return count;
    }


    /**
     * 825. 适龄的朋友(计数排序 + 前缀和)
     * @param ages
     * @return
     */
    public int numFriendRequests__(int[] ages) {
        int[] cnt = new int[121];
        for (int age:ages){
            ++cnt[age];
        }
        int[] pre = new int[121];
        for (int i = 1; i <=120; i++){
            pre[i] = pre[i-1]+cnt[i];
        }
        int count = 0;
        for (int i = 15; i <= 120;i++){
            if(cnt[i] > 0){
                int bound = (int)(i*0.5+8);
                count+=cnt[i]*(pre[i] - pre[bound-1] - 1);
            }
        }
        return count;
    }


    /**
     * 930. 和相同的二元子数组（暴力）
     * @param nums
     * @param goal
     * @return
     */
    public int numSubarraysWithSum(int[] nums, int goal) {
        int n = nums.length;
        int[] prefix = new int[n+1];
        int count = 0;
        for (int i = 1; i <=n;i++){
            prefix[i] = prefix[i-1]+nums[i-1];
        }
        for (int i = 0; i < n;i++){
            for (int j = i+1; j <= n;j++){
                if(prefix[j]-prefix[i] == goal){
                    count++;
                }
            }
        }
        return count;
    }

    /**
     * 930. 和相同的二元子数组（哈希表）
     * @param nums
     * @param goal
     * @return
     */
    public int numSubarraysWithSum_(int[] nums, int goal) {
        HashMap<Integer,Integer> map = new HashMap<>();
        int sum = 0;
        int ret = 0;
        for (int num:nums){
            map.put(sum, map.getOrDefault(sum,0)+1);
            sum+=num;
            ret+=map.getOrDefault(sum-goal,0);
        }
        return ret;
    }

    /**
     * 1004. 最大连续1的个数 III
     * @param nums
     * @param k
     * @return
     */
    public int longestOnes(int[] nums, int k) {
        int n = nums.length;
        if(k == n){
            return k;
        }
        int[] prefix = new int[n+1];
        int ret = 0;
        for (int i = 1; i <= n; i++){
            prefix[i] = prefix[i-1]+nums[i-1];
        }
        for (int i = 0; i<n; i++){
            for (int j=i+1; j<=n; j++){
                if (prefix[j]-prefix[i]+k==j-i){
                    ret = Math.max(ret,j-i);
                }
            }
        }
        return ret;
    }

    /**
     * 1004. 最大连续1的个数 III
     * @param nums
     * @param k
     * @return
     */
    public int longestOnes_(int[] nums, int k) {
        int n = nums.length;
        int[] prefix = new int[n+1];
        int ret = 0;
        for (int i = 1; i <= n; i++){
            prefix[i] = prefix[i-1]+(1-nums[i-1]);
        }
        for (int i = 0; i<n; i++){
            int left = binarySearch(prefix,prefix[i+1]-k);
            ret = Math.max(ret, i-left+1);
        }
        return ret;
    }

    private int binarySearch(int[] prefix, int target){
        int low = 0; int high = prefix.length-1;
        while (low < high){
            int mid = (high+low)/2;
            if(prefix[mid] < target){
                low=mid+1;
            }else {
                high=mid;
            }
        }
        return low;
    }


    /**
     * 1154. 一年中的第几天
     * @param date
     * @return
     */
    public int dayOfYear(String date) {
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd");
        Date date1 = null;
        try {
            date1 = sdf.parse(date);

        }catch (ParseException e){
            e.printStackTrace();
        }
        String str = String.format("%tj", date1);
        return Integer.parseInt(str);
    }

    /**
     * 1154. 一年中的第几天
     * @param date
     * @return
     */
    public long dayOfYear_(String date) {
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd");
        Date date1 = null;
        Calendar ca = null;
        try {
            date1 = sdf.parse(date);
            ca = Calendar.getInstance();
            ca.setTime(date1);

        }catch (ParseException e){
            e.printStackTrace();
        }
        return ca.get(Calendar.DAY_OF_YEAR);
    }

    /**
     * 304. 二维区域和检索 - 矩阵不可变
     */
    class NumMatrix {
        int[][] matrix;
        int[][] pre_matrix_sum;
        public NumMatrix(int[][] matrix) {
            this.matrix = matrix;
            int n = this.matrix.length;
            int m = this.matrix[0].length;
            pre_matrix_sum = new int[n+1][m+1];
            for (int i = 1;i < n+1;i++){
                for (int j = 1;j < m+1;j++){
                    pre_matrix_sum[i][j] = pre_matrix_sum[i-1][j]+pre_matrix_sum[i][j-1]
                            +this.matrix[i-1][j-1]-pre_matrix_sum[i-1][j-1];
                }
            }
        }

        public int sumRegion(int row1, int col1, int row2, int col2) {
            return pre_matrix_sum[row2+1][col2+1]-pre_matrix_sum[row2+1][col1]
                    -pre_matrix_sum[row1][col2+1]
                    +pre_matrix_sum[row1][col1];
        }
    }


    /**
     * 303. 区域和检索 - 数组不可变
     */
    class NumArray {
        int[] pre_nums;
        public NumArray(int[] nums) {
            int n = nums.length;
            pre_nums = new int[n+1];
            for (int i = 1;i < n+1;i++){
                pre_nums[i] = nums[i-1]+pre_nums[i-1];
            }
        }

        public int sumRange(int left, int right) {
            return pre_nums[right]-pre_nums[left-1];
        }
    }

    /**
     * 363. 矩形区域不超过 K 的最大数值和
     * @param matrix
     * @param k
     * @return
     */
    public int maxSumSubmatrix(int[][] matrix, int k) {
        int n = matrix.length;
        int m = matrix[0].length;
        int[][] pre_sum = new int[n+1][m+1];
        for (int i = 1; i <= n; i++){
            for (int j = 1; j <= m; j++){
                pre_sum[i][j] = pre_sum[i-1][j]+pre_sum[i][j-1]-pre_sum[i-1][j-1]+matrix[i-1][j-1];
            }
        }
        int max = Integer.MIN_VALUE;
        for (int i = 0; i <= n;i++){
            for (int j = 0; j <= m;j++){
                for (int p = n; p > i;p--){
                    for (int q = m; q > j;q--){
                        int sum = pre_sum[p][q]-pre_sum[p][j]-pre_sum[i][q]+pre_sum[i][j];
                        if(sum <= k){
                            max = Math.max(max,sum);
                        }
                    }
                }
            }
        }
        return max;
    }

    /**
     * 437. 路径总和 III(递归)
     */
    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode() {}
        TreeNode(int val) { this.val = val; }
        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }

    public int pathSum(TreeNode root, int targetSum) {
        if(root == null){
            return 0;
        }
        int path_nums = 0;
        path_nums = rootSum(root, targetSum);
        path_nums+=pathSum(root.left,targetSum);
        path_nums+=pathSum(root.right,targetSum);

        return path_nums;
    }

    public int rootSum(TreeNode root, int targetSum){
        int nums = 0;
        if(root == null){
            return 0;
        }
        if(root.val == targetSum){
            nums++;
        }
        nums += rootSum(root.left, targetSum-root.val);
        nums += rootSum(root.right, targetSum-root.val);
        return nums;
    }

    /**
     * 437. 路径总和 III(前缀和)
     * @param root
     * @param targetSum
     * @return
     */
    public int pathSum_(TreeNode root, int targetSum) {
        if(root == null){
            return 0;
        }
        HashMap<Long,Integer> prefix = new HashMap<>();
        prefix.put(0L,1);
        return dfs(root,prefix,0,targetSum);
    }

    public int dfs(TreeNode root,HashMap<Long,Integer> prefix, long curr, int targetSum){
        if(root == null){
            return 0;
        }
        curr += root.val;
        int nums = prefix.getOrDefault(curr-targetSum,0);
        prefix.put(curr, prefix.getOrDefault(curr,0)+1);
        nums += dfs(root.left,prefix,curr,targetSum);
        nums += dfs(root.right,prefix,curr,targetSum);
        /**
         * 一些细节：由于我们只能统计往下的路径，
         * 但是树的遍历会同时搜索两个方向的子树。
         * 因此我们应当在搜索完以某个节点为根的左右子树之后，
         * 应当回溯地将路径总和从哈希表中删除，防止统计到跨越两个方向的路径。
         */
        prefix.put(curr, prefix.getOrDefault(curr,0)-1);
        return nums;
    }

    /**
     * 523. 连续的子数组和(超时)
     * @param nums
     * @param k
     * @return
     */
    public boolean checkSubarraySum(int[] nums, int k) {
        int n = nums.length;
        int[] prefixs = new int[n+1];
        for (int i = 1; i <=n; i++){
            prefixs[i] += prefixs[i-1]+nums[i-1];
        }
        boolean flag = false;
        for (int i=n;i>=2;i--){
            for (int j=0;j<i;j++){
                int subSum = prefixs[i]-prefixs[j];
                if((i-j)>=2 && subSum%k==0){
                    flag = true;
                }
            }
        }
        return flag;
    }

    public boolean checkSubarraySum_(int[] nums, int k) {
        int n = nums.length;
        HashMap<Integer,Integer> map = new HashMap<>();
        int[] prefixs = new int[n+1];
        for (int i = 1; i <=n; i++){
            prefixs[i] += prefixs[i-1]+nums[i-1];

        }
        for(int i = 0; i <=n; i++){
            if(map.containsKey(prefixs[i]%k)){
                if(i-map.get(prefixs[i]%k) >= 2) {
                    return true;
                }
            }else {
                map.put(prefixs[i]%k,i);
            }
        }
        return false;
    }


    /**
     * 1208. 尽可能使字符串相等(滑动窗口)
     */
    public int equalSubstring(String s, String t, int maxCost) {
        if(s == null || t == null){
            return 0;
        }
        int m = s.length();
        int sum = 0;
        int max_len = 0;
        for (int right=0, left = 0; left < m&&right < m;){
            sum+=Math.abs(s.charAt(right)-t.charAt(right));
            if(sum <= maxCost){
                max_len = Math.max(max_len, right-left+1);
                right++;
            }else {
                sum-=Math.abs(s.charAt(left)-t.charAt(left));
                left++;
                right++;
            }
        }
        return max_len;
    }

    /**
     * 1208. 尽可能使字符串相等（滑动窗口修改版）
     */
    public int equalSubstring_(String s, String t, int maxCost) {
        if(s == null || t == null){
            return 0;
        }
        int m = s.length();
        int sum = 0;
        int max_len = 0;
        int left =0, right = 0;
        while(right < m){
            sum+=Math.abs(s.charAt(right)-t.charAt(right));
            while (sum > maxCost){
                sum-=Math.abs(s.charAt(left)-t.charAt(left));
                left++;
            }
            max_len = Math.max(max_len, right-left+1);
            right++;
        }
        return max_len;
    }


    /**
     * 1208. 尽可能使字符串相等（前缀和+二分查找）
     */
    public int equalSubstring__(String s, String t, int maxCost) {
        if(s == null || t == null){
            return 0;
        }
        int n = s.length();
        int max_len = 0;
        int[] prefix_sum = new int[n+1];
        for (int i = 1; i <=n; i++){
            prefix_sum[i] = prefix_sum[i-1]+(s.charAt(i-1)-t.charAt(i-1));
        }
        for (int i = 1; i <=n; i++){
            int left = binarySearch(prefix_sum, i, prefix_sum[i]-maxCost);
            max_len = Math.max(max_len, i-left);
        }
        return max_len;
    }

    private int binarySearch(int[] prefix, int right, int target){
        int low = 0;
        int high = right;
        while (low < high){
            int mid = (low+high)/2;
            if(prefix[mid] < target){
                low = mid+1;
            }else {
                high = mid;
            }
        }
        return low;
    }

    /**
     * 1310. 子数组异或查询
     */
    public int[] xorQueries(int[] arr, int[][] queries) {
        int n = arr.length;
        int m = queries.length;
        int[] prefix_xor = new int[n+1];
        int[] results = new int[m];
        for (int i = 1; i <= n; i++){
            prefix_xor[i] = prefix_xor[i-1]^arr[i-1];
        }
        for (int i = 0; i < m; i++){
            results[i] = prefix_xor[queries[i][1]+1]^prefix_xor[queries[i][0]];
        }
        return results;
    }

    /**
     * 1442. 形成两个异或相等数组的三元组数目（三重循环）
     */
    public int countTriplets(int[] arr) {
        int n = arr.length;
        int num=0;
        int[] prefix_xor = new int[n+1];
        for (int i=1;i<=n;i++){
            prefix_xor[i]=prefix_xor[i-1]^arr[i-1];
        }
        for (int i=0;i<n-1;i++){
            for (int j=i+1;j<n;j++){
                for (int k=j;k<n;k++){
                    if(prefix_xor[i]==prefix_xor[k+1]){
                        num++;
                    }
                }
            }
        }
        return num;
    }

    /**
     * 1442. 形成两个异或相等数组的三元组数目（二重循环）
     */
    public int countTriplets_(int[] arr) {
        int n = arr.length;
        int num=0;
        int[] prefix_xor = new int[n+1];
        for (int i=1;i<=n;i++){
            prefix_xor[i]=prefix_xor[i-1]^arr[i-1];
        }
        for (int i=0;i<n-1;i++){
            for (int k=i+1;k<n;k++){
                if(prefix_xor[i]==prefix_xor[k+1]){
                    num++;
                }
            }
        }
        return num;
    }

    /**
     * 1442. 形成两个异或相等数组的三元组数目（哈希表）
     */
    public int countTriplets__(int[] arr) {
        int n = arr.length;
        int num=0;
        int[] prefix_xor = new int[n+1];
        for (int i=1;i<=n;i++){
            prefix_xor[i]=prefix_xor[i-1]^arr[i-1];
        }
        HashMap<Integer, Integer> cnt = new HashMap<>();
        HashMap<Integer, Integer> total = new HashMap<>();
        for (int k=0;k<n;++k){
            if(cnt.containsKey(prefix_xor[k+1])){
                num+=cnt.get(prefix_xor[k+1])*k-total.get(prefix_xor[k+1]);
            }
            cnt.put(prefix_xor[k],cnt.getOrDefault(prefix_xor[k],0)+1);
            total.put(prefix_xor[k],total.getOrDefault(prefix_xor[k],0)+k);
        }
        return num;
    }

    /**
     * 1480. 一维数组的动态和
     */
    public int[] runningSum(int[] nums) {
        int n = nums.length;
        int[] prefix = new int[n];
        for (int i=0;i<n;i++){
            if(i==0){
                prefix[i]=nums[i];
            }else{
                prefix[i]=prefix[i-1]+nums[i];
            }
        }
        return prefix;
    }


    /**
     * 1588. 所有奇数长度子数组的和
     */
    public int sumOddLengthSubarrays(int[] arr) {
        int n=arr.length;
        int[] prefix=new int[n+1];
        for (int i=1;i<=n;i++){
            prefix[i]=prefix[i-1]+arr[i-1];
        }
        int sum=0;
        for (int i=1;i<=n;i++){
            for (int k=i;k<=n;k++){
                if((k-i+1)%2!=0){
                  sum+=prefix[k]-prefix[i-1];
                }
            }
        }
        return sum;
    }


    /**
     * 1738. 找出第 K 大的异或坐标值
     */
    public int kthLargestValue(int[][] matrix, int k) {
        int n = matrix.length;
        int m = matrix[0].length;
        int[][] prefix_m=new int[n+1][m+1];
        int[] eyes=new int[n*m];
        int index=0;
        for (int i=1;i<=n;i++){
            for (int j=1;j<=m;j++){
                prefix_m[i][j]=prefix_m[i-1][j]^prefix_m[i][j-1]^prefix_m[i-1][j-1]^matrix[i-1][j-1];
                eyes[index++]=prefix_m[i][j];
            }
        }
        Arrays.sort(eyes);
        return eyes[n*m-k];
    }

////TODO
    /**
     * 1744. 你能在你最喜欢的那天吃到你最喜欢的糖果吗？
     */

    /**
     * 1749. 任意子数组和的绝对值的最大值（暴力超时）
     */
    public int maxAbsoluteSum(int[] nums) {
        if(nums == null){
            return 0;
        }
        int n = nums.length;
        if (n==0){
            return 0;
        }
        int result = Integer.MIN_VALUE;
        for (int i = 0; i < n; i++){
            for (int j = 0; j < n; j++){
                int prefix_num = subnumsprefix_sum(i,j,nums);
                result = Math.max(prefix_num, result);
            }
        }
        return result;
    }
    private int subnumsprefix_sum(int i, int j, int[] nums){
        int prefix_sum = 0;
        for (int k = i; k <= j; k++){
            prefix_sum = prefix_sum+nums[k];
        }
        return Math.abs(prefix_sum);
    }

    /**
     * 1838. 最高频元素的频数
     */
    public int maxFrequency(int[] nums, int k) {
        Arrays.sort(nums);
        int n = nums.length;
        int total = 0;
        int l = 0;
        int max = 1;
        for (int r = 1; r < n;r++){
            total += (nums[r]-nums[r-1])*(r-l);
            while (total > k && l < r){
                total -= (nums[r]-nums[l++]);
            }
            max = Math.max(max, r-l+1);

        }
        return max;
    }


    /**
     * 1037. 有效的回旋镖
     * 在二维空间中，叉乘还有另一个几何意义。 叉积等于由向量a和向量b构成的平行四边形的面积
     */
    public boolean isBoomerang(int[][] points) {
        int[] a = {points[1][0]-points[0][0], points[1][1]-points[0][1]};
        int[] b = {points[2][0]-points[0][0], points[2][1]-points[0][1]};
        return (a[0]*b[1]-b[0]*a[1]) != 0;
    }

    /**
     * 560. 和为 K 的子数组（暴力）
     */
    public int subarraySum(int[] nums, int k) {
        int n = nums.length;
        int count = 0;
        for (int l = 0; l < n; l++){
            for (int r = l; r < n; r++){
                int sum = 0;
                for (int i = l; i <= r; i++){
                    sum+=nums[i];
                }
                if(sum == k){
                    count++;
                }
            }
        }
        return count;
    }

    /**
     * 560. 和为 K 的子数组（前缀和）
     */
    public int subarraySum_prefix(int[] nums, int k) {
        int n = nums.length;
        int count = 0;
        int[] prefix = new int[n+1];
        for (int i = 1; i <= n; i++){
            prefix[i]=prefix[i-1]+nums[i-1];
        }
        for (int l = 0; l < n; l++){
            for (int r = l+1; r <= n; r++){
                int sum = prefix[r]-prefix[l];
                if(sum ==k){
                    count++;
                }
            }
        }
        return count;
    }

    /**
     * 560. 和为 K 的子数组（枚举）
     */
    public int subarraySum_enum(int[] nums, int k) {
        int n = nums.length;
        int count = 0;
        for (int i = 0; i < n; i++){
            int sum = 0;
            for (int j = i; j >=0; j--){
                sum+=nums[j];
                if(sum == k){
                    count++;
                }
            }
        }
        return count;
    }

    /**
     * 560. 和为 K 的子数组（前缀和+哈希表）
     */
    public int subarraySum_prefix_hash(int[] nums, int k) {
        int n = nums.length;
        int count = 0;
        HashMap<Integer, Integer> map = new HashMap<>();
        int[] prefix = new int[n+1];
        for (int i = 1; i <= n; i++){
            prefix[i]=prefix[i-1]+nums[i-1];
        }
        map.put(0,1);
        for (int r = 1; r <= n; r++){
            if (map.containsKey(prefix[r]-k)){
                count+=map.get(prefix[r]-k);
            }
            map.put(prefix[r], map.getOrDefault(prefix[r],0)+1);
        }
        return count;
    }


    /**
     * 497. 非重叠矩形中的随机点
     */
    class Rectangle{

        int[][] rects = null;
        int[] sum = null;
        int n = 0;
        Random random = new Random();
        public Rectangle(int[][] rects) {
            this.rects = rects;
            int n = rects.length;
            sum = new int[n+1];
            for (int i = 1; i <= n; i++){
                sum[i] = sum[i-1]+(rects[i-1][2]-rects[i-1][0]+1)*(rects[i-1][3]-rects[i-1][1]+1);
            }
        }
        public int[] pick() {
            int target = random.nextInt(sum[n])+1;
            int i = binarySearch(sum,target);
            int[] rect = rects[i-1];
            int x = rect[0]+random.nextInt(rect[2]-rect[0]+1);
            int y = rect[1]+random.nextInt(rect[3]-rect[1]+1);
            return new int[]{x,y};
        }

        public int binarySearch(int[] arrs, int target){
            int l = 0, r = n;
            while (l < r) {
                int mid = l + r >> 1;
                if (arrs[mid] >= target) r = mid;
                else l = mid + 1;
            }
            return r;
        }
    }

    /**
     * 剑指 Offer 47. 礼物的最大价值
     */
    public int maxValue(int[][] grid) {
        if(grid.length == 0) return 0;
        int m = grid.length;
        int n = grid[0].length;
        for (int i = 0; i < m; i++){
            for (int j = 0; j < n; j++){
                if (i == 0 && j == 0) continue;
                if (i == 0) grid[i][j]=grid[i][j-1]+grid[i][j];
                else if (j == 0) grid[i][j]=grid[i-1][j]+grid[i][j];
                else grid[i][j]=grid[i][j]+Math.max(grid[i-1][j], grid[i][j-1]);
            }
        }
        return grid[m-1][n-1];
    }


    /**
     * 187. 重复的DNA序列(哈希表)
     */
    public List<String> findRepeatedDnaSequences(String s) {
        if(s == null || s.length() <= 10) return new ArrayList<>();
        int n = s.length();
        HashMap<String, Integer> map = new HashMap<>();
        for (int i = 0; i+10 <= n; i++){
            map.put(s.substring(i, i+10), map.getOrDefault(s.substring(i, i+10), 0)+1);
        }
        List<String> list = new ArrayList<String>();
        for (Map.Entry<String, Integer> entry : map.entrySet()){
            if(entry.getValue() > 1){
                list.add(entry.getKey());
            }
        }
        return list;
    }


    /**
     * 730. 统计不同回文子序列
     */
    public long countPalindromicSubsequences(String s) {
        final int MOD = 1000000007;
        int n = s.length();
        long[][] dp = new long[n][n];
        for (int i = 0; i < n;i++){
            dp[i][i]=1L;
        }
        for (int len = 2; len <= n; len++){
            for (int i = 0; i+len <= n; i++){
                int j = i+len-1;
                if (s.charAt(i) == s.charAt(j)){
                    int left = i+1;
                    int right = j-1;
                    while (left <= right && s.charAt(left) != s.charAt(i)){
                        left++;
                    }
                    while (left <= right && s.charAt(right) != s.charAt(j)){
                        right--;
                    }
                    if(left > right){
                        dp[i][j] = 2*dp[i+1][j-1]+2;
                    } else if (left == right) {
                        dp[i][j] = 2*dp[i+1][j-1]+1;
                    }else {
                        dp[i][j] = 2*dp[i+1][j-1] - dp[left+1][right-1];
                    }
                }else {
                    dp[i][j] = dp[i+1][j]+dp[i][j-1]-dp[i+1][j-1];
                }
                dp[i][j] = (dp[i][j] >= 0) ? dp[i][j] % MOD : dp[i][j];
            }
        }
        return dp[0][n-1];
    }


    /**
     * 324. 摆动排序 II
     */
    public void wiggleSort(int[] nums) {
        int n = nums.length;
        int x = (n+1)/2;
        Arrays.sort(nums);
        int[] numsb = nums.clone();
        for (int i = 0,j=x-1, k=n-1; i < n; i+=2, k--, j--){
            nums[i] = numsb[j];
            if(i+1 < n){
                nums[i+1]=numsb[k];
            }
        }
    }

    /**
     * 1252. 奇数值单元格的数目
     */
    public int oddCells(int m, int n, int[][] indices) {
        int[][] matrix = new int[m][n];
        int len = indices.length;
        for (int i = 0; i < len; i++){
            int row = indices[i][0];
            int col = indices[i][1];
            for (int j = 0; j < n; j++){
                matrix[row][j]++;
            }
            for (int k = 0; k < m; k++){
                matrix[k][col]++;
            }
        }
        int count = 0;
        for (int i = 0; i < m; i++){
            for (int j =0; j < n; j++){
                if (matrix[i][j]%2!=0){
                    count++;
                }
            }
        }
        return count;
    }

    /**
     * 735. 行星碰撞
     */
    public int[] asteroidCollision(int[] asteroids) {
        Stack<Integer> stack = new Stack<>();
        for (int aster: asteroids){
            boolean alive = true;
            while (alive && aster < 0 && !stack.isEmpty()&& stack.peek() > 0){
                alive = stack.peek() < -aster;
                if (stack.peek() <= -aster){
                    stack.pop();
                }
            }
            if (alive){
                stack.push(aster);
            }
        }
        int len = stack.size();
        int[] results = new int[len];
        for (int i = len-1; i >= 0; i--){
            results[i] = stack.pop();
        }
        return results;
    }

    public boolean isSameSigns(int a, int b){
        return (a ^ b >>> 31) == 0;
    }

    /**
     * 1408. 数组中的字符串匹配
     */
    public List<String> stringMatching(String[] words) {
        int len = words.length;
        Set<String> results = new HashSet<>();
        for (int i = 0; i < len; i++){
            for (int j = 0; j < len; j++){
                if (j == i) continue;
                if (words[i].contains(words[j])){
                    results.add(words[j]);
                }
            }
        }
        return new ArrayList<>(results);
    }

    /**
     * 745. 前缀和后缀搜索
     */
    class WordFilter {
        Map<String,Integer> dictionary;
        public WordFilter(String[] words) {
            dictionary = new HashMap<>();
            for (int i = 0; i < words.length; i++){
                String word = words[i];
                int m = word.length();
                for (int prefixLength = 1; prefixLength <= m; prefixLength++){
                    for (int suffixLength = 1; suffixLength <= m; suffixLength++){
                        dictionary.put(word.substring(0,prefixLength) + "#" +word.substring(m-suffixLength), i);
                    }
                }
            }
        }

        public int f(String pref, String suff) {
            return dictionary.getOrDefault(pref+"#"+suff, -1);
        }
    }

    /**
     * 1625. 执行操作后字典序最小的字符串
     */
    public String findLexSmallestString(String s, int a, int b) {
        System.out.println(rotation(s,b));
        return "";
    }

    private String rotation(String s, int b){
        char[] chars = s.toCharArray();
        int n = chars.length;
        char[] chars1 = new char[n];
        for (int i = 0; i < n; i++){
            chars1[(i+b)%n] = chars[i];
        }
        return new String(chars1);
    }

    private String accumulation(String s, int a){
        int n = s.length();
        char[] chars = new char[n];
        for (int i = 1; i < n; i+=2){
            char c = chars[i];
            int num = (int)(c-'0');
            chars[i] = (char) ((a+num)%10 + '0');
        }
        return new String(chars);
    }

    /**
     * 558. 四叉树交集
     */
    class QuadNode {
        public boolean val;
        public boolean isLeaf;
        public QuadNode topLeft;
        public QuadNode topRight;
        public QuadNode bottomLeft;
        public QuadNode bottomRight;

        public QuadNode() {}
        public QuadNode(boolean _val, boolean _isLeaf){
            val=_val;
            isLeaf = _isLeaf;
        }

        public QuadNode(boolean _val,boolean _isLeaf,QuadNode _topLeft,QuadNode _topRight,QuadNode _bottomLeft,QuadNode _bottomRight) {
            val = _val;
            isLeaf = _isLeaf;
            topLeft = _topLeft;
            topRight = _topRight;
            bottomLeft = _bottomLeft;
            bottomRight = _bottomRight;
        }
    };
    public QuadNode intersect(QuadNode quadTree1, QuadNode quadTree2) {
        if(quadTree1.isLeaf){
            if (quadTree1.val){
                return new QuadNode(true, true);
            }
            return new QuadNode(quadTree2.val, quadTree2.isLeaf, quadTree2.topLeft, quadTree2.topRight, quadTree2.bottomLeft, quadTree2.bottomRight);
        }
        if (quadTree2.isLeaf){
            intersect(quadTree2, quadTree1);
        }
        QuadNode node1 = intersect(quadTree1.topLeft, quadTree2.topLeft);
        QuadNode node2 = intersect(quadTree1.topRight, quadTree2.topRight);
        QuadNode node3 = intersect(quadTree1.bottomLeft, quadTree2.bottomLeft);
        QuadNode node4 = intersect(quadTree1.bottomRight, quadTree2.bottomRight);
        if (node1.isLeaf && node2.isLeaf && node3.isLeaf && node4.isLeaf
        && node1.val==node2.val && node1.val==node3.val && node1.val==node4.val){
            return new QuadNode(node1.val, true);
        }
        return new QuadNode(false, false, node1, node2, node3, node4);
    }


    /**
     * 剑指 Offer II 041. 滑动窗口的平均值
     */
    class MovingAverage {
        int[] numbers;
        int index;
        int size;
        int sum = 0;

        /** Initialize your data structure here. */
        public MovingAverage(int size) {
            numbers = new int[size];
            this.size = size;
        }

        public double next(int val) {
            if (index >= size){
                sum-=numbers[index%size];
            }
            numbers[index%size] = val;
            sum+=val;
            index++;
            if (index >= size){
                return (double) sum/size;
            }
            return (double) sum/index;
        }
    }

    /**
     * 565. 数组嵌套(超时)
     */
    public int arrayNesting(int[] nums) {
        int n = nums.length;
        int result = 0;
        for (int i = 0; i < n;i++){
            List<Integer> list = new ArrayList<>();
            addNesting(list, nums, i);
            result = Math.max(result, list.size());
        }
        return result;
    }

    private void addNesting(List<Integer> list, int[] nums, int i){
        if (list.contains(nums[i])){
            return;
        }
        list.add(nums[i]);
        int index = nums[i];
        addNesting(list, nums, index);
    }

    /**
     * 565. 数组嵌套(图)
     */
    public int arrayNesting_(int[] nums) {
        int n = nums.length;
        int result = 0;
        for (int i = 0; i < n;i++){
            int count = 0;
            while (nums[i] < n){
                int num = nums[i];
                nums[i] = n;
                i = num;
                count++;
            }
            result = Math.max(result, count);
        }
        return result;
    }


    /**
     * 749. 隔离病毒
     */
    static int[][] dirs = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

    public int containVirus(int[][] isInfected) {
        int m = isInfected.length, n = isInfected[0].length;
        int ans = 0;
        while (true) {
            List<Set<Integer>> neighbors = new ArrayList<Set<Integer>>();
            List<Integer> firewalls = new ArrayList<Integer>();
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (isInfected[i][j] == 1) {
                        Queue<int[]> queue = new ArrayDeque<int[]>();
                        queue.offer(new int[]{i, j});
                        Set<Integer> neighbor = new HashSet<Integer>();
                        int firewall = 0, idx = neighbors.size() + 1;
                        isInfected[i][j] = -idx;

                        while (!queue.isEmpty()) {
                            int[] arr = queue.poll();
                            int x = arr[0], y = arr[1];
                            //上下左右
                            for (int d = 0; d < 4; ++d) {
                                int nx = x + dirs[d][0], ny = y + dirs[d][1];
                                if (nx >= 0 && nx < m && ny >= 0 && ny < n) {
                                    if (isInfected[nx][ny] == 1) {
                                        queue.offer(new int[]{nx, ny});
                                        isInfected[nx][ny] = -idx;
                                    } else if (isInfected[nx][ny] == 0) {
                                        ++firewall;
                                        neighbor.add(getHash(nx, ny));
                                    }
                                }
                            }
                        }
                        neighbors.add(neighbor);
                        firewalls.add(firewall);
                    }
                }
            }

            if (neighbors.isEmpty()) {
                break;
            }

            int idx = 0;
            for (int i = 1; i < neighbors.size(); ++i) {
                if (neighbors.get(i).size() > neighbors.get(idx).size()) {
                    idx = i;
                }
            }
            ans += firewalls.get(idx);
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (isInfected[i][j] < 0) {
                        if (isInfected[i][j] != -idx - 1) {
                            isInfected[i][j] = 1;
                        } else {
                            isInfected[i][j] = 2;
                        }
                    }
                }
            }
            for (int i = 0; i < neighbors.size(); ++i) {
                if (i != idx) {
                    for (int val : neighbors.get(i)) {
                        int x = val >> 16, y = val & ((1 << 16) - 1);
                        isInfected[x][y] = 1;
                    }
                }
            }
            if (neighbors.size() == 1) {
                break;
            }
        }
        return ans;
    }

    public int getHash(int x, int y) {
        return (x << 16) ^ y;
    }


    /**
     * 731. 我的日程安排表 II
     */
    class MyCalendarTwo {
        List<int[]> booked;
        List<int[]> overlaps;

        public MyCalendarTwo() {
            booked = new ArrayList<int[]>();
            overlaps = new ArrayList<int[]>();
        }

        public boolean book(int start, int end) {
            for (int[] arr : overlaps) {
                int l = arr[0], r = arr[1];
                if (l < end && start < r) {
                    return false;
                }
            }
            for (int[] arr : booked) {
                int l = arr[0], r = arr[1];
                if (l < end && start < r) {
                    overlaps.add(new int[]{Math.max(l, start), Math.min(r, end)});
                }
            }
            booked.add(new int[]{start, end});
            return true;
        }
    }

    /**
     * 1455. 检查单词是否为句中其他单词的前缀
     * @param sentence
     * @param searchWord
     * @return
     */
    public int isPrefixOfWord(String sentence, String searchWord) {
        if(sentence == null || sentence.equals("")) return -1;
        if (searchWord == null || searchWord.equals("")) return -1;
        String[] words = sentence.split(" ");
        int len = searchWord.length();
        for (int i = 0; i < words.length; i++){
            if (words[i].length() >= len){
                String preWord = words[i].substring(0,len);
                if (preWord.equals(searchWord)){
                    return i+1;
                }
            }
        }
        return -1;
    }


    /**
     * 2. 两数相加
     */
    public static class ListNode {
        int val;
        ListNode next;
        ListNode() {}
        ListNode(int val) { this.val = val; }
        ListNode(int val, ListNode next) { this.val = val; this.next = next; }
    }
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode l3 = l1;
        if(l1 == null){
            return l2;
        }
        if(l2 == null){
            return l1;
        }
        int tmp = 0;
        ListNode pre = l1;
        while (l1 != null && l2 != null){
            pre = l1;
            l1.val+=tmp;
            if(l1.val+l2.val>=10){
                l1.val = (l2.val+l1.val)-10;
                tmp = 1;
            }else{
                l1.val = l1.val+l2.val;
                tmp = 0;
            }
            l1 = l1.next;
            l2 = l2.next;
        }
        while (l2 != null){
            pre.next = l2;
            if (l2.val+tmp >= 10){
                l2.val = (l2.val+tmp)-10;
                tmp = 1;
            }else {
                l2.val = l2.val+tmp;
                tmp = 0;
            }
            pre = l2;
            l2 = l2.next;
        }
        while (l1 != null){
            pre = l1;
            if (l1.val+tmp >= 10){
                l1.val = (l1.val+tmp)-10;
                tmp = 1;
            }else {
                l1.val = l1.val+tmp;
                tmp = 0;
            }
            l1 = l1.next;
        }
        if (tmp != 0){
            pre.next = new ListNode(tmp);
        }
        return l3;

    }

    /**
     * 655. 输出二叉树
     * @param root
     * @return
     */
    public List<List<String>> printTree(TreeNode root) {
        //格式化布局
        ///计算树高
        int height = calDepth(root);
        ///行m
        int m = height+1;
        ///列n
        int n = (1 << (height+1))-1;
        ///初始化
        List<List<String>> res = new ArrayList<List<String>>();
        for (int i = 0; i < m; i++){
            List<String> row = new ArrayList<>();
            for (int j = 0; j < n; j++){
                row.add("");
            }
            res.add(row);
        }
        dfs(res, root, 0, (n-1)/2, height);
        return res;
    }

    public int calDepth(TreeNode root){
        int h = 0;
        if(root.left != null){
            h = Math.max(h, calDepth(root.left)+1);
        }
        if(root.right != null){
            h = Math.max(h, calDepth(root.right)+1);
        }
        return h;
    }

    public void dfs(List<List<String>> res, TreeNode root, int r, int c, int height){
        res.get(r).set(c, Integer.toString(root.val));
        if (root.left != null){
            dfs(res, root.left, r+1, c-(1 << height-r-1), height);
        }
        if (root.right != null){
            dfs(res, root.right, r+1, c+(1 << height-r-1), height);
        }
    }

    /**
     * 3. 无重复字符的最长子串
     */
    public int lengthOfLongestSubstring(String s) {
        int n = s.length();
        Set<Character> set = new HashSet<>();
        int rk = -1, ans = 0;
        for (int i = 0; i < n; i++){
            if (i != 0){
                set.remove(s.charAt(i-1));
            }
            while (rk+1 < n && !set.contains(s.charAt(rk+1))){
                set.add(s.charAt(rk+1));
                rk++;
            }
            ans = Math.max(ans, rk-i+1);
        }
        return ans;
    }


    /**
     * 1460. 通过翻转子数组使两个数组相等
     */
    public boolean canBeEqual(int[] target, int[] arr) {
        Arrays.sort(target);
        Arrays.sort(arr);
        int n = target.length;
        for (int i = 0; i < n; i++){
            if (target[i] != arr[i]){
                return false;
            }
        }
        return true;
    }

    /**
     *5. 最长回文子串(暴力 超时)
     */
    public String longestPalindrome(String s) {
        int n = s.length();
        String ans = "";
        for (int i = 1; i <= n; i++){
            for (int j = 0; j < n-i+1; j++){
                String subString = s.substring(j, j+i);
                if (isPalindrome(subString)){
                    ans = subString;
                }
            }
        }
        return ans;
    }

    public boolean isPalindrome(String s){
        int n = s.length();
        for (int i = 0; i < n/2; i++){
            if (s.charAt(i) != s.charAt(n-1-i)) return false;
        }
        return true;
    }


    /**
     * 5. 最长回文子串 (动态规划：P(i,j) = P(i+1, j-1) ^ (Si == Sj))
     */
    public String longestPalindrome_dp(String s) {
        int len = s.length();
        if (len < 2){
            return s;
        }
        int maxLen = 1;
        int begin = 0;
        boolean[][] dp = new boolean[len][len];
        for (int i = 0; i < len; i++){
            dp[i][i] = true;
        }

        char[] charArray = s.toCharArray();
        for (int L = 2; L <= len; L++){
            for (int i = 0; i < len; i++){
                int j = L+i-1;
                if(j >= len){
                    break;
                }
                if (charArray[i] != charArray[j]){
                    dp[i][j] = false;
                }else {
                    if(j - i < 3){
                        dp[i][j] = true;
                    }else {
                        dp[i][j] = dp[i+1][j-1];
                    }
                }

                if (dp[i][j] && j-i+1 > maxLen){
                    maxLen = j-i+1;
                    begin = 1;
                }
            }
        }
        return s.substring(begin, begin+maxLen);
    }

    /**
     * 946. 验证栈序列
     */
    public boolean validateStackSequences(int[] pushed, int[] popped) {
        Deque<Integer> stack = new ArrayDeque<>();
        int n = pushed.length;
        for (int i = 0, j = 0; i < n; i++){
            stack.push(pushed[i]);
            while (!stack.isEmpty() && stack.peek() == popped[j]){
                stack.pop();
                j++;
            }
        }
        return stack.isEmpty();
    }

    /**
     * 998. 最大二叉树 II
     */
    public TreeNode insertIntoMaxTree(TreeNode root, int val) {
        TreeNode parent = null;
        TreeNode cur = root;
        while (cur != null){
            if (val > cur.val){
                if (parent == null){
                    return new TreeNode(val, root, null);
                }else {
                    TreeNode node = new TreeNode(val, cur, null);
                    parent.right = node;
                    return root;
                }
            }else {
                parent = cur;
                cur = cur.right;
            }
        }
        parent.right = new TreeNode(val);
        return root;
    }

    /**
     * 1636. 按照频率将数组升序排序
     */
    public int[] frequencySort(int[] nums) {
        int n = nums.length;
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < n; i++){
            map.put(nums[i],map.getOrDefault(nums[i],0)+1);
        }
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < n; i++){
            list.add(nums[i]);
        }
        Collections.sort(list,(a,b)->{
            int countA = map.get(a),countB = map.get(b);
            return countA != countB ? countA - countB : b - a;
        });
        int[] ans = new int[n];
        for (int i = 0; i < n; i++){
            ans[i] = list.get(i);
        }
        return ans;
    }

    /**
     * 698. 划分为k个相等的子集
     */
    public boolean canPartitionKSubsets(int[] nums, int k) {
        int n = nums.length;
        int sum = 0;
        for (int i = 0; i < n; i++){
            sum+=nums[i];
        }
        if(sum % k != 0){
            return false;
        }
        Arrays.sort(nums);
        boolean[] used = new boolean[n];
        int part = sum/k;
        for (int i = 0; i < n; i++){
            used[i] = true;
            int index = binarySearch4Part(nums,part-nums[i],used);
            if (index == -1){
                return false;
            }
            used[index] = true;
        }
        return true;
    }

    private int binarySearch4Part(int[] nums,int key,boolean[] used){
        int start = 0;
        int end = nums.length;
        while (start <= end){
            int mid = (end+start)/2;
            if(nums[mid] == key && !used[mid]){
                return mid;
            }else if(nums[mid] < key) {
               start = mid+1;
            }else {
                end = mid-1;
            }
        }
        return -1;
    }


    /**
     * 9. 回文数
     */
    public boolean isPalindrome(int x) {
        if(x < 0 || (x % 10 == 0 && x != 0)) return false;
        int revertedNumber = 0;
        while (x > revertedNumber){
            revertedNumber = revertedNumber * 10 + x % 10;
            x /= 10;
        }
        return x == revertedNumber || x == revertedNumber/10;
    }

    /**
     * 13. 罗马数字转整数
     */
    public int romanToInt(String s) {
        int res = 0;
        Map<Character,Integer> map = new HashMap<>();
        map.put('I',1);
        map.put('V',5);
        map.put('X',10);
        map.put('L',50);
        map.put('C',100);
        map.put('D',500);
        map.put('M',1000);
        int n = s.length();
        for (int i = 0; i < n; i++){
            int value = map.get(s.charAt(i));
            if (i < n-1 && value < map.get(s.charAt(i+1))){
                res-=map.get(s.charAt(i));
            }else{
                res+=map.get(s.charAt(i));
            }
        }
        return res;
    }


    public int arraySign(int[] nums) {
        int product = 1;
        for (int num : nums) {
            if ( num == 0) return 0;
            product *= Integer.compare(num, 0);
        }
        return Integer.compare(product, 0);
    }

    /**
     * 704. 二分查找
     */
    public int search(int[] nums, int target) {
        int left = 0;
        int right = nums.length-1;
        while (left <= right){
            int mid = (left+right)/2;
            if (nums[mid] < target){
                left = mid+1;
            }else if (nums[mid] > target){
                right = mid-1;
            }else {
                return mid;
            }
        }
        return -1;
    }

    /**
     * 35. 搜索插入位置
     */
    public int searchInsert(int[] nums, int target) {
        int left = 0;
        int right = nums.length-1;
        while (left <= right){
            int mid = (left+right)/2;
            if (nums[mid] < target){
                left = mid+1;
            }else if (nums[mid] > target){
                right = mid-1;
            }else {
                return mid;
            }
        }
        return left;
    }

    /**
     * 34. 在排序数组中查找元素的第一个和最后一个位置(暴力)
     */
    public int[] searchRange(int[] nums, int target) {
        int n = nums.length;
        int[] res = new int[2];
        int position = -1;
        for (int i = 0; i < n; i++){
            if (nums[i] == target){
                position = i;
                break;
            }
        }
        if (position == -1){
            return new int[]{-1,-1};
        }
        res[0] = position;
        while (position < n-1){
            if (nums[position+1] == target){
                position += 1;
            }else{
                break;
            }
        }
        res[1] = position;
        return res;
    }


    /**
     * 34. 在排序数组中查找元素的第一个和最后一个位置(二分查找)
     */
    public int[] searchRange_(int[] nums, int target) {
        int left = getLeftBorder(nums,target);
        int right = getRightBorder(nums,target);
        if (left == -2 || right == -2) return new int[]{-1,-1};
        if (right - left > 1) return new int[]{left+1,right-1};
        return new int[]{-1,-1};
    }

    /**
     * left找到的是最右target位置+1
     * @param nums
     * @param target
     * @return
     */
    private int getRightBorder(int[] nums, int target){
        int left = 0;
        int right = nums.length - 1;
        int rightBorder = -2;
        while (left <= right){
            int mid = (left+right)/2;
            if(nums[mid] > target){
                right = mid - 1;
            }else {
                left = mid + 1;
                rightBorder = left;//直到找到
            }
        }
        return rightBorder;
    }

    /**
     * right找到的是target最左位置-1
     * @param nums
     * @param target
     * @return
     */
    private int getLeftBorder(int[] nums, int target){
        int left = 0;
        int right = nums.length - 1;
        int leftBorder = -2;
        while (left <= right){
            int mid = (left+right)/2;
            if(nums[mid] < target){
                left = mid + 1;
            }else {
                right = mid - 1;
                leftBorder = right;
            }
        }
        return leftBorder;
    }


    /**
     * 69. x 的平方根(牛顿迭代法)
     */
    public int mySqrt(int x) {
        if (x == 0) return 0;
        double x0 = x;
        while (true){
            double xi = 0.5 * (x0 + (double) x /x0);
            if (Math.abs(x0 - xi) < 1e-7){
                break;
            }
            x0 = xi;
        }
        return (int) x0;
    }

    /**
     * 69. x 的平方根(二分查找)
     */
    public int mySqrt_(int x) {
        int l = 0, r = x; int ans = -1;
        while (l <= r){
            int mid = (l+r)/2;
            if ((long)mid * mid <= x){
                ans = mid;
                l = mid+1;
            }else {
                r = mid-1;
            }
        }
        return ans;
    }

    /**
     * 367. 有效的完全平方数
     */
    public boolean isPerfectSquare(int num) {
        int l = 0, r = num;
        while (l <= r){
            int mid = (l+r)/2;
            if ((long) mid*mid == num){
                return true;
            }else if ((long) mid * mid < num){
                l = mid+1;
            }else {
                r = mid-1;
            }
        }
        return false;
    }

    /**
     * 27. 移除元素(双指针)
     */
    public int removeElement(int[] nums, int val) {
        int left = 0;
        int n = nums.length;
        for (int right = 0;right < n; right++){
            if (nums[right] != val){
                nums[left] = nums[right];
                left++;
            }
        }
        return left;
    }

    /**
     * 27. 移除元素（双指针优化）
     */
    public int removeElement_(int[] nums, int val) {
        int left = 0;
        int right = nums.length;

        while (left < right){
            if (nums[left] == val){
                nums[left] = nums[right -1];
                right--;
            }else {
                left++;
            }
        }
        return left;
    }

    /**
     * 27. 移除元素（相向双指针）
     */
    public int removeElement__(int[] nums, int val) {
        int left = 0;
        int right = nums.length-1;
        //将不等于val的值移动到左边
        while (left <= right){
            while (left <= right && nums[left] != val){
                left++;
            }
            while (left <= right && nums[right] == val){
                right--;
            }
            if (left < right){
                nums[left++] = nums[right--];
            }
        }
        return left;
    }

    /**
     * 26.删除排序数组中的重复项
     */
    public int removeDuplicates(int[] nums) {
        int i = 0, j = 0;
        int n = nums.length-1;
        while (j <= n) {
            if (nums[i] == nums[j]){
                j++;
            }else{
                nums[i+1] = nums[j];
                i++;
            }
        }
        return i+1;
    }


    /**
     * 283.移动零
     */
    public void moveZeroes(int[] nums) {
        int n=nums.length-1;
        int i=0,j=0;
        while (j <= n){
            if(nums[j] != 0){
                nums[i]=nums[j];
                i++;
            }
            j++;
        }
        while(i <= n){
            nums[i]=0;
            i++;
        }
    }

    /**
     * 844. 比较含退格的字符串
     */
    public boolean backspaceCompare(String s, String t) {
        Stack<Character> sStack = new Stack<>();
        Stack<Character> tStack = new Stack<>();
        for (char c:s.toCharArray()){
            if (c == '#' && !sStack.isEmpty()){
                sStack.pop();
            }else if (c != '#'){
                sStack.push(c);
            }
        }
        for (char c:t.toCharArray()){
            if (c == '#' && !tStack.isEmpty()){
                tStack.pop();
            }else if (c != '#'){
                tStack.push(c);
            }
        }
        if (sStack.size() != tStack.size()){
            return false;
        }

        while (!sStack.isEmpty() && !tStack.isEmpty()){
            if (sStack.pop() != tStack.pop()) return false;
        }
        return true;
    }

    /**
     * 977. 有序数组的平方
     */
    public int[] sortedSquares(int[] nums) {
        int n = nums.length;
        for (int i = 0; i < n; i++){
            nums[i] = nums[i]*nums[i];
        }
        Arrays.sort(nums);
        return nums;
    }

    /**
     * 977. 有序数组的平方（双指针）
     */
    public int[] sortedSquares_(int[] nums) {
        int n = nums.length;
        int neg = -1;
        for (int i = 0; i < n; i++){
            if (nums[i] < 0){
                neg = i;
            }else{
                break;
            }
        }
        int[] ans = new int[n];
        int index = 0, i = neg, j = neg+1;
        while (i >= 0 || j < n){
            if (i < 0){
                ans[index] = nums[j]*nums[j];
                j++;
            } else if (j == n) {
                ans[index] = nums[i]*nums[i];
                --i;
            } else if (nums[i]*nums[i] < nums[j]*nums[j]) {
                ans[index] = nums[i]*nums[i];
                --i;
            }else {
                ans[index] = nums[j]*nums[j];
                j++;
            }
            ++index;
        }
        return ans;
    }

    /**
     * 209. 长度最小的子数组（暴力，超时）
     */
    public int minSubArrayLen(int target, int[] nums) {
        int n = nums.length;
        int[] prefix_nums = new int[n+1];
        for (int i = 1; i <= n; i++){
            prefix_nums[i] = nums[i-1]+prefix_nums[i-1];
        }
        int minLen = Integer.MAX_VALUE;
        for (int i = 0; i < n; i++){
            for (int j = n; j > i; j--){
                if (prefix_nums[j]-prefix_nums[i]>=target){
                    minLen = Math.min(minLen,j-i);
                }
            }
        }
        return minLen == Integer.MAX_VALUE?0:minLen;
    }

    /**
     * 209. 长度最小的子数组（前缀和+滑动窗口）
     */
    public int minSubArrayLen_(int target, int[] nums) {
        int n = nums.length;
        int[] prefix_nums = new int[n+1];
        for (int i = 1; i <= n; i++){
            prefix_nums[i] = nums[i-1]+prefix_nums[i-1];
        }
        int minLen = Integer.MAX_VALUE;
        int i = 0;
        for (int j = 1; j <= n;){
            if (prefix_nums[j]-prefix_nums[i] >= target){
                minLen = Math.min(minLen,j-i);
                i++;
            }else {
                j++;
            }
        }
        return minLen == Integer.MAX_VALUE?0:minLen;
    }

    /**
     * 209. 长度最小的子数组（前缀和+二分查找）
     */
    public int minSubArrayLen__(int target, int[] nums) {
        int n = nums.length;
        int[] prefix_nums = new int[n+1];
        for (int i = 1; i <= n; i++){
            prefix_nums[i] = nums[i-1]+prefix_nums[i-1];
        }
        int minLen = Integer.MAX_VALUE;
        for (int j = 1; j <= n;j++){
            int s = target+prefix_nums[j-1];
            int bound = Arrays.binarySearch(prefix_nums,s);
            if (bound < 0){
                bound = -bound-1;
            }
            if (bound <= n){
                minLen = Math.min(minLen, bound-(j-1));
            }
        }
        return minLen == Integer.MAX_VALUE?0:minLen;
    }


    /**
     * 209. 长度最小的子数组（滑动窗口）
     */
    public int minSubArrayLen___(int target, int[] nums) {
        int n = nums.length;
        int sums = 0;
        int minLen = Integer.MAX_VALUE;
        int i = 0;
        for (int j = 0; j < n;j++){
            sums+=nums[j];
            while (sums >= target){
                minLen = Math.min(minLen,j-i+1);

                sums -= nums[i];
                i++;
            }

        }
        return minLen == Integer.MAX_VALUE?0:minLen;
    }

    /**
     * 904. 水果成篮(滑动窗口)
     */
    public int totalFruit(int[] fruits) {
        int n = fruits.length;
        Map<Integer, Integer> map = new HashMap<>();
        int left = 0, ans = 0;
        for (int right = 0; right < n; right++){
            map.put(fruits[right],map.getOrDefault(fruits[right],0)+1);
            while (map.size() > 2){
                map.put(fruits[left],map.get(fruits[left])-1);
                if (map.get(fruits[left]) == 0){
                    map.remove(fruits[left]);
                }
                ++left;
            }
            ans = Math.max(ans,right-left+1);
        }
        return ans;
    }

    /**
     * 76. 最小覆盖子串(滑动窗口)
     */
    public String minWindow(String s, String t) {
        if(s == null || t == null){
            return null;
        }
        if ("".equals(t)){
            return "";
        }
        int sLen = s.length();
        int tLen = t.length();
        char[] charArrayS = s.toCharArray();
        char[] charArrayT = t.toCharArray();

        int[] tFreq = new int[128];
        int[] winFreq = new int[128];
        for (char c:charArrayT){
            tFreq[c]++;
        }
        int distance = 0;
        int minLen = sLen+1;
        int left = 0, right = 0;
        int begin = 0;
        while (right < sLen){
            if (tFreq[charArrayS[right]] == 0){
                right++;
                continue;
            }
            if (winFreq[charArrayS[right]] < tFreq[charArrayS[right]]){
                distance++;
            }
            winFreq[charArrayS[right]]++;
            right++;
            while (distance == tLen){
                if(right - left < minLen){
                    minLen = right - left;
                    begin = left;
                }
                if (tFreq[charArrayS[left]] == 0){
                    left++;
                    continue;
                }
                if (winFreq[charArrayS[left]] == tFreq[charArrayS[left]]){
                    distance--;
                }
                winFreq[charArrayS[left]]--;
                left++;
            }
        }
        if (minLen == sLen+1){
            return "";
        }
        return s.substring(begin,begin+minLen);
    }

    /**
     * 练习：思考下列问题为什么可以使用【滑动窗口】
     * 第3题：无重复字符的最长字串（可以用Map加快计算）
     * 第209题：长度最小的子数组
     * 第424题：替换后的最长重复字符
     * 第438题：找到字符串中所有字母异位词
     * 第567题：字符串的排列
     * 理论化名字：决策单调性
     */

}
