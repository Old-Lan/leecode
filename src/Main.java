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
        int[] nums = {2,-5,1,-4,3,-2};
        System.out.println(main.maxAbsoluteSum(nums));
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





}
