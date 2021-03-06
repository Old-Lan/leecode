
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
        int[] nums = {5,4,0,3,1,6,2};
        System.out.println(main.arrayNesting(nums));


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
     * 2100. ???????????????????????????(??????)
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
     * 2100. ???????????????????????????(????????????)
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
     * 2100. ???????????????????????????(?????????)
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
            non[i] = security[i - 1] < security[i] ? 1 : -1;//1??????????????????-1???????????????
        }
        int[] nonincre = new int[n + 1];//???????????????
        int[] nondecre = new int[n + 1];//???????????????
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
     * 525. ????????????(??????)
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
     * 525. ????????????(?????????+??????)
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
     * 528. ?????????????????????
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
     * 724. ???????????????????????????
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
     * 825. ???????????????(??????)
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
     * 825. ???????????????(??????+?????????)
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
     * 825. ???????????????(???????????? + ?????????)
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
     * 930. ???????????????????????????????????????
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
     * 930. ??????????????????????????????????????????
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
     * 1004. ????????????1????????? III
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
     * 1004. ????????????1????????? III
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
     * 1154. ?????????????????????
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
     * 1154. ?????????????????????
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
     * 304. ????????????????????? - ???????????????
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
     * 303. ??????????????? - ???????????????
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
     * 363. ????????????????????? K ??????????????????
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
     * 437. ???????????? III(??????)
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
     * 437. ???????????? III(?????????)
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
         * ?????????????????????????????????????????????????????????
         * ?????????????????????????????????????????????????????????
         * ???????????????????????????????????????????????????????????????????????????
         * ???????????????????????????????????????????????????????????????????????????????????????????????????
         */
        prefix.put(curr, prefix.getOrDefault(curr,0)-1);
        return nums;
    }

    /**
     * 523. ?????????????????????(??????)
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
     * 1208. ???????????????????????????(????????????)
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
     * 1208. ??????????????????????????????????????????????????????
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
     * 1208. ???????????????????????????????????????+???????????????
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
     * 1310. ?????????????????????
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
     * 1442. ??????????????????????????????????????????????????????????????????
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
     * 1442. ??????????????????????????????????????????????????????????????????
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
     * 1442. ???????????????????????????????????????????????????????????????
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
     * 1480. ????????????????????????
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
     * 1588. ?????????????????????????????????
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
     * 1738. ????????? K ?????????????????????
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
     * 1744. ???????????????????????????????????????????????????????????????
     */

    /**
     * 1749. ????????????????????????????????????????????????????????????
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
     * 1838. ????????????????????????
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
     * 1037. ??????????????????
     * ????????????????????????????????????????????????????????? ?????????????????????a?????????b?????????????????????????????????
     */
    public boolean isBoomerang(int[][] points) {
        int[] a = {points[1][0]-points[0][0], points[1][1]-points[0][1]};
        int[] b = {points[2][0]-points[0][0], points[2][1]-points[0][1]};
        return (a[0]*b[1]-b[0]*a[1]) != 0;
    }

    /**
     * 560. ?????? K ????????????????????????
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
     * 560. ?????? K ???????????????????????????
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
     * 560. ?????? K ????????????????????????
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
     * 560. ?????? K ????????????????????????+????????????
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
     * 497. ??????????????????????????????
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
     * ?????? Offer 47. ?????????????????????
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
     * 187. ?????????DNA??????(?????????)
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
     * 730. ???????????????????????????
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
     * 324. ???????????? II
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
     * 1252. ???????????????????????????
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
     * 735. ????????????
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
     * 1408. ???????????????????????????
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
     * 745. ?????????????????????
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
     * 1625. ??????????????????????????????????????????
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
     * 558. ???????????????
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
     * ?????? Offer II 041. ????????????????????????
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
     * 565. ????????????(??????)
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
     * 565. ????????????(???)
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
     * 749. ????????????
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
                            //????????????
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
     * 731. ????????????????????? II
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





}
