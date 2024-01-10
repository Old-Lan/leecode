
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.regex.Pattern;

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
//        String s = "bbbextm";
//        String t = "bbb#extm";
//        System.out.println(main.backspaceCompare(s,t));
//        int[] nums = {-1};
//        System.out.println(Arrays.toString(main.sortedSquares_(nums)));
//        int target = 7;
//        int[] nums = {2,3,1,2,4,3};
//        System.out.println(main.minSubArrayLen___(target,nums));
//        int[] fruits = {1,2,1};
//        System.out.println(main.totalFruit(fruits));
//        String s = "a";
//        String t = "a";
//        System.out.println(main.minWindow(s,t));
//        int n = 4;
//        System.out.println(Arrays.deepToString(main.generateMatrix(n)));
//        int[][] matrix = {{1,2,3,4},{5,6,7,8},{9,10,11,12}};
//        System.out.println(main.spiralOrder(matrix));

//        ListNode node6_ = new ListNode(7,node3);
        ListNode node6 = new ListNode(6,null);
        ListNode node5 = new ListNode(5,node6);
        ListNode node4 = new ListNode(4,node5);
        ListNode node3 = new ListNode(3,node4);
        ListNode node2 = new ListNode(2,node3);
        ListNode node1 = new ListNode(1,node2);
//        node6.next = node2;
        main.printListNode(node1);
        ListNode head = main.removeElements(node1,1);
        main.printListNode(head);
//        System.out.println();
//        main.printListNode(node1);
//        ListNode head = main.removeNthFromEnd(node1,1);
//        main.printListNode(head);
//        System.out.println(main.detectCycle(node1).val);
//        String s = "a";
//        String t = "ab";
//        System.out.println(main.isAnagram(s,t));
//        int[] nums1 = {4,9,5};
//        int[] nums2 = {9,4,9,8,4};
//        System.out.println(Arrays.toString(main.intersection(nums1, nums2)));
//        String ransomNote = "aab";
//        String magazine = "baa";
//        System.out.println(main.canConstruct(ransomNote,magazine));
//        String[] strs = {"eat","tea","tan","ate","nat","bat"};
//        System.out.println(main.groupAnagrams(strs));
//        String s = "cbaebabacd";
//        String s = "abab";
//        String p = "ab";
//        System.out.println(main.findAnagrams(s,p));
//        int[] nums1 = {4,9,5};
//        int[] nums2 = {9,4,9,8,4};
//        System.out.println(Arrays.toString(main.intersect(nums1, nums2)));
//        System.out.println(81*3);
//        int[] nums = {1,2,3,4,4,9,56,90};
//        int target = 8;
//        System.out.println(Arrays.toString(main.twoSum(nums, target)));
//        int[] nums1 = {1,2};
//        int[] nums2 = {-2,-1};
//        int[] nums3 = {-1,2};
//        int[] nums4 = {0,2};
//        System.out.println(main.fourSumCount(nums1,nums2,nums3,nums4));
//        int[] nums = {1000000000,1000000000,1000000000,1000000000};
//        System.out.println(main.fourSum(nums,-294967296));
//        System.out.println(Integer.MIN_VALUE);
//        String s = "abcd";
//        int k = 2;
//        System.out.println(main.reverseStr(s,k));
//        String words = "  Bob    Loves  Alice   ";
//        System.out.println(main.reverseWords(words));
//        String s = "abcdefg";
//        int n = 2;
//        System.out.println(main.reverseLeftWords(s,n));
//        String haystack = "abc";
//        String needle = "c";
//        System.out.println(main.strStr(haystack,needle));
//        String s = "aabaaba";
//        System.out.println(main.repeatedSubstringPattern(s));
//        System.out.printf("%5d %d",1,2,3);
//        System.out.printf("%5d %f", 1);
//        System.out.printf("%5d %f",1,2);
//        System.out.printf("%.2f\n%0.2f", 1.23456, 0.0);
//        System.out.printf("%08s\n","Java");
//        System.out.printf("%05d %06.1f\n",32,32.32);
//        Scanner scanner = new Scanner(System.in);
//        String ssn = scanner.nextLine();
//        String pattern = "\\d\\d\\d-\\d\\d-\\d\\d\\d\\d";
//        boolean isMatch = Pattern.matches(pattern,ssn);
//        System.out.println(isMatch)
//        System.out.println(main.climbStairs(3));
//        int[] cost = {10,15,20};
//        main.minCostClimbingStairs(cost);
//        int[] nums = {847,847,0,0,0,399,416,416,879,879,206,206,206,272};
//        int[] results = main.applyOperations(nums);
//        main.printArray(results);

//        int[] nums = {2,1,3,0,6};
//        int[] results = main.selectSort(nums);
//        main.printArray(results);
//        int m = 3;
//        int n = 7;
//        System.out.println(main.uniquePaths(3,3));
//        int[][] obstacleGrid = {{0,0},{1,1},{0,0}};
//        System.out.println(main.uniquePathsWithObstacles(obstacleGrid));
//        int n = 3;
//        System.out.println(main.numTrees(n));
//        String[] strs = {"flower","flow","flight"};
//        String[] strs = {"dog","racecar","car"};
//        String[] strs = {"a"};
//        System.out.println(main.longestCommonPrefix(strs));
//        int[] arr = {5,7,4,5,8,1,6,0,3,4,6,1,7};
//        System.out.println(win1(arr));
//        System.out.println(win2(arr));
//        System.out.println(win3(arr));
//        int[] weights = {3,2,4,7,3,1,7};
//        int[] values = {5,6,3,19,12,4,2};
//        int bag = 15;
//        System.out.println(maxValue(weights,values,bag));
//        System.out.println(dp(weights,values,bag));

//        char a = '0'+1;
//        System.out.println(a);
//        Myqueue myqueue = new Myqueue();
//        myqueue.push(1);
//        myqueue.push(2);
//        myqueue.pop();
//        myqueue.push(3);
//        System.out.println(myqueue.pop());
//        int[][] obstacleGrid = {{0,0,0},{0,1,0},{0,0,0}};
//
//        int result = main.uniquePathsWithObstacles(obstacleGrid);
//        System.out.println(result);
//        ListNode a1 = new ListNode(3);
//        ListNode a2 = new ListNode(2);
//        ListNode c1 = new ListNode(0);
//        ListNode c2 = new ListNode(-4);
//        ListNode c3 = new ListNode(5);
//        ListNode b1 = new ListNode(5);
//        ListNode b2 = new ListNode(0);
//        ListNode b3 = new ListNode(1);
//        a1.next = a2;
//        a2.next = c1;
//        c1.next = c2;
//        c2.next = c3;
//        c3.next = c2;
//        b1.next = b2;
//        b2.next = b3;
//        b3.next = c1;
//        ListNode cycleListNode = main.detectCycle(a1);
//        System.out.println(cycleListNode.val);
//        int[] nums = {0,0,0};
//        List<List<Integer>> results = main.threeSum_brute_repeat(nums);
//        System.out.println(results);
    }

    public Main(){
    }


    public int[] twoSum__(int[] nums, int target) {
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
    public boolean backspaceCompare_(String s, String t) {
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
     * 844. 比较含退格的字符串
     */
    public boolean backspaceCompare(String s, String t) {
        int i = s.length()-1, j = t.length()-1;
        int skipS = 0, skipT = 0;
        while (i >= 0 || j >= 0){
            while (i >= 0){
                if (s.charAt(i) == '#'){
                    skipS++;
                    i--;
                } else if (skipS > 0) {
                    skipS--;
                    i--;
                }else {
                    break;
                }
            }
            while (j >= 0){
                if (t.charAt(j) == '#'){
                    skipT++;
                    j--;
                } else if (skipT > 0) {
                    skipT--;
                    j--;
                }else {
                    break;
                }
            }
            if (i >= 0 && j >= 0){
                if (s.charAt(i) != t.charAt(j)){
                    return false;
                }
            }else {
                if (i >= 0 || j >= 0){
                    return false;
                }
            }
            i--;
            j--;
        }
        return true;
    }

    /**
     * 977. 有序数组的平方
     */
    public int[] sortedSquares__(int[] nums) {
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
     * 977. 有序数组的平方
     */
    public int[] sortedSquares(int[] nums) {
//        int n = nums.length;
//        for (int i = 1; i < n; i++){
//            int tmp = nums[i]*nums[i];
//            for (int j = i-1; j >= 0; --j){
//                if (nums[j]>tmp) nums[j+1]
//            }
//        }
        return nums;
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


    /**
     * 59. 螺旋矩阵 II（模拟）
     */
    public int[][] generateMatrix_(int n) {
        int maxNum = n * n;
        int curNum = 1;
        int[][] matrix = new int[n][n];
        int row = 0, column = 0;
        int[][] directions = {{0,1},{1,0},{0,-1},{-1,0}};//右下左上
        int directionIndex = 0;
        while (curNum <= maxNum){
            matrix[row][column] = curNum;
            curNum++;
            int nextRow = row+directions[directionIndex][0], nextColumn = column+directions[directionIndex][1];//判断下一个方向
            if(nextRow < 0 || nextRow >= n || nextColumn < 0 || nextColumn >= n || matrix[nextRow][nextColumn] != 0){
                directionIndex = (directionIndex + 1) % 4;//顺时针旋转至下一个方向
            }
            row = row+directions[directionIndex][0];
            column = column+directions[directionIndex][1];
        }
        return matrix;
    }


    /**
     * 59. 螺旋矩阵 II（按层模拟）
     */
    public int[][] generateMatrix(int n) {
        int curNum = 1;
        int[][] matrix = new int[n][n];
        int left = 0, right = n-1, top = 0, bottom = n-1;
        while (left <= right && top <= bottom){
            for (int column = left; column <= right; column++){
                matrix[top][column] = curNum++;
            }
            for (int row = top+1; row <= bottom; row++){
                matrix[row][right] = curNum++;
            }

            if (left < right && top < bottom){
                for (int column = right-1; column > left; column--){
                    matrix[bottom][column] = curNum++;
                }
                for (int row = bottom; row > top; row--){
                    matrix[row][left] = curNum++;
                }
            }
            left++;
            right--;
            top++;
            bottom--;
        }
        return matrix;
    }

    /**
     * 54. 螺旋矩阵
     */
    public List<Integer> spiralOrder(int[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;
        boolean[][] flags = new boolean[m][n];
        int[][] directions = {{0,1},{1,0},{0,-1},{-1,0}};
        int currentIndex = 1;
        int directionIndex = 0;
        int row = 0, column = 0;
        List<Integer> results = new ArrayList<>();
        while (currentIndex <= m*n){
            results.add(matrix[row][column]);
            flags[row][column] = true;
            currentIndex++;
            int nextRow = row+directions[directionIndex][0], nextColumn = column+directions[directionIndex][1];
            if (nextColumn < 0 || nextColumn >= n || nextRow < 0 || nextRow >= m || flags[nextRow][nextColumn]){
                directionIndex = (directionIndex + 1) % 4;
            }
            row = row+directions[directionIndex][0];
            column = column+directions[directionIndex][1];
        }
        return results;
    }

    /**
     * 203. 移除链表元素
     */
    public ListNode removeElements___(ListNode head, int val) {
        if(head == null) return null;
        ListNode L = new ListNode(-1,head);
        ListNode tmp = L;
        while (tmp.next != null){
            if (tmp.next.val == val){
                tmp.next = tmp.next.next;
            }else{
                tmp = tmp.next;
            }
        }
        return L.next;
    }


    /**
     * 203. 移除链表元素(复习)
     * @param head
     * @param val
     * @return
     */
    public ListNode removeElements__(ListNode head, int val){
        //考虑删除第一个节点和其他节点的处理方式相同，所以需要一个头结点
        ListNode L = new ListNode(-1,head);
        ListNode tmp = L;
        while (tmp.next != null){
            if (tmp.next.val == val){
                tmp.next = tmp.next.next;
            }else {
                tmp = tmp.next;
            }
        }
        return L.next;
    }


    /**
     * 203. 移除链表元素（递归）
     */
    public ListNode removeElements_(ListNode head, int val) {
        if(head == null) return null;
        head.next = removeElements_(head.next,val);
        return head.val == val ? head.next:head;
    }

    /**
     * 203. 移除链表元素（递归-复习）
     * @param head
     * @param val
     * @return
     */
    public ListNode removeElements(ListNode head, int val){
        if (head == null){
            return null;
        }
        head.next = removeElements(head.next,val);
        return head.val == val ? head.next:head;
    }

    public void printListNode(ListNode head){
        while (head != null){
            System.out.print(head.val+",");
            head = head.next;
        }
        System.out.println();
    }

    /**
     * 707.设计链表
     */
    class MyLinkedList {

        int size;
        ListNode head;
        public MyLinkedList() {
            size = 0;
            head = new ListNode(0);
        }

        public int get(int index) {
            if(index < 0 || index >= size) return -1;
            ListNode cur = head;
            for (int i = 0; i <= index; i++){
                cur = cur.next;
            }
            return cur.val;
        }

        public void addAtHead(int val) {
            addAtIndex(0,val);
        }

        public void addAtTail(int val) {
            addAtIndex(size,val);
        }

        public void addAtIndex(int index, int val) {
            if(index > size) return;
            index = Math.max(0,index);
            size++;
            ListNode insertNode = new ListNode(val);
            ListNode cur = head;
            for (int i = 0; i <= index-1; i++){
                cur = cur.next;
            }
            insertNode.next = cur.next;
            cur.next = insertNode;
        }

        public void deleteAtIndex(int index) {
            if (index < 0 || index >= size) return;
            size--;
            ListNode cur = head;
            for (int i = 0; i <= index-1; i++){
                cur = cur.next;
            }
            cur.next = cur.next.next;
        }
    }


    /**
     * 206. 反转链表(迭代)
     */
    public ListNode reverseList(ListNode head) {
        if(head == null) return null;
        ListNode pre = null;
        ListNode cur = head;
        while (cur != null){
            ListNode tmp = cur.next;
            cur.next = pre;
            pre = cur;
            cur = tmp;

        }
        return pre;
    }


    /**
     * 206. 反转链表(递归)
     */
    //1->2->3->4->5->null
    public ListNode reverseList_(ListNode head) {
        if(head == null || head.next == null){
            return head;
        }
        ListNode newHead = reverseList_(head.next);
        head.next.next = head;
        head.next = null;
        return newHead;
    }

    /**
     * 24. 两两交换链表中的节点
     */
    public ListNode swapPairs_(ListNode head) {
        if (head == null){
            return null;
        }
        ListNode pre = head;
        while (pre != null && pre.next != null){
            ListNode next = pre.next;
            int tmp = pre.val;
            pre.val = next.val;
            next.val = tmp;
            pre = pre.next.next;
        }
        return head;
    }

    /**
     * 24. 两两交换链表中的节点
     */
    public ListNode swapPairs(ListNode head) {
        if (head == null){
            return null;
        }
        ListNode L = new ListNode(-1);
        L.next = head;
        ListNode pre = L;
        while (pre.next != null && pre.next.next!= null){
            ListNode next = pre.next.next;
            ListNode tmp = pre.next;
            tmp.next = next.next;
            pre.next = next;
            next.next = tmp;
            pre = pre.next.next;
        }
        return L.next;
    }

    /**
     * 19. 删除链表的倒数第 N 个结点
     */
    public ListNode removeNthFromEnd_(ListNode head, int n) {
        if (head == null) return null;
        if (n < 0) return head;
        ListNode L = new ListNode(-1);
        L.next = head;
        ListNode cur = L;
        int size = 0;
        while (head != null){
            size++;
            head = head.next;
        }
        for (int i = 0; i < size-n-1; i++){
            cur = cur.next;
        }
        cur.next = cur.next.next;
        return L.next;
    }

    /**
     * 19. 删除链表的倒数第 N 个结点
     */
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(0, head);
        ListNode first = head;
        ListNode second = dummy;
        for (int i = 0; i < n; ++i){
            first = first.next;
        }
        while (first != null){
            first = first.next;
            second = second.next;
        }
        second.next = second.next.next;
        return dummy.next;
    }

    /**
     * 面试题 02.07. 链表相交
     */
    public ListNode getIntersectionNode_(ListNode headA, ListNode headB) {
        Set<ListNode> visited = new HashSet<>();
        ListNode temp = headA;
        while (temp != null){
            visited.add(temp);
            temp = temp.next;
        }
        temp = headB;
        while (temp != null){
            if (visited.contains(temp)){
                return temp;
            }
            temp = temp.next;
        }
        return null;
    }

    /**
     * 142. 环形链表 II
     */
    public ListNode detectCycle_(ListNode head) {
        Set<ListNode> visited = new HashSet<>();
        while (head != null){
            if (visited.contains(head)){
                return head;
            }
            visited.add(head);
            head = head.next;
        }
        return null;
    }

    /**
     * 142. 环形链表 II
     */
    public ListNode detectCycle(ListNode head) {
        ListNode fast = head;
        ListNode slow = head;
        while(fast != null && fast.next != null){
            fast = fast.next.next;
            slow = slow.next;
            if (slow == fast){//第一次相遇
                ListNode pre = head;
                while (pre != slow){//往前走到入口处
                    pre = pre.next;
                    slow = slow.next;
                }
                return pre;
            }
        }
        return null;
    }

    /**
     * 242. 有效的字母异位词
     */
    public boolean isAnagram(String s, String t) {
        Map<Character,Integer> mapS = new HashMap<>();
        for (char c:s.toCharArray()){
            mapS.put(c,mapS.getOrDefault(c,0)+1);
        }
        Map<Character,Integer> mapT = new HashMap<>();
        for (char c:t.toCharArray()){
            mapT.put(c,mapT.getOrDefault(c,0)+1);
        }
        if (mapS.size() != mapT.size()) return false;
        for (Character key:mapS.keySet()){
            if (!Objects.equals(mapS.get(key), mapT.getOrDefault(key, 0))){
                return false;
            }
        }
        return true;
    }

    /**
     * 349. 两个数组的交集
     */
    public int[] intersection_(int[] nums1, int[] nums2) {
        Set<Integer> set1 = new HashSet<>();
        Set<Integer> set2 = new HashSet<>();
        for (int k : nums1) {
            set1.add(k);
        }
        for (int j : nums2) {
           set2.add(j);
        }
        List<Integer> list = new ArrayList<>();
        for (int key : set1){
            if (!set2.contains(key)) continue;
            list.add(key);
        }
        int len = list.size();
        int[] array = new int[len];
        for (int i=0; i < len; i++){
            array[i] = list.get(i);
        }
        return array;
    }

    /**
     * 349. 两个数组的交集（排序+双指针）
     */
    public int[] intersection(int[] nums1, int[] nums2) {
        Arrays.sort(nums1);
        Arrays.sort(nums2);
        int len1 = nums1.length;
        int len2 = nums2.length;
        int[] results = new int[len1+len2];
        int index1 = 0, index2 = 0, index3 = 0;
        while (index1 < len1 && index2 < len2){
            int num1 = nums1[index1];
            int num2 = nums2[index2];
            if (num1 == num2){
                if (index3 == 0 || results[index3-1] != num1){
                    results[index3++] = num1;
                }
                index1++;
                index2++;
            } else if (num1 < num2) {
                index1++;
            }else {
                index2++;
            }
        }
        return Arrays.copyOf(results, index3);
    }

    /**
     * 383. 赎金信
     */
    public boolean canConstruct(String ransomNote, String magazine) {
        Map<Character,Integer> map = new HashMap<>();
        for (char c: magazine.toCharArray()){
            map.put(c,map.getOrDefault(c,0)+1);
        }
        for (char c:ransomNote.toCharArray()){
            map.put(c,map.getOrDefault(c,0)-1);
            if (map.get(c) < 0){
                return false;
            }
        }
        return true;
    }


    /**
     * 49. 字母异位词分组（排序）
     */
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> map = new HashMap<>();
        for (String str:strs){
            char[] chars = str.toCharArray();
            Arrays.sort(chars);
            String key = new String(chars);
            List<String> list = map.getOrDefault(key, new ArrayList<>());
            list.add(str);
            map.put(key, list);
        }
        return new ArrayList<>(new ArrayList<>(map.values()));
    }

    /**
     * 438. 找到字符串中所有字母异位词
     */
    public List<Integer> findAnagrams(String s, String p) {
        int lenS = s.length();
        int lenP = p.length();
        List<Integer> results = new ArrayList<>();
        if (lenS < lenP) return results;
        int[] sCount = new int[26];
        int[] pCount = new int[26];
        for (int i = 0; i < lenP; i++){
            ++sCount[s.charAt(i)-'a'];
            ++pCount[p.charAt(i)-'a'];
        }
        if (Arrays.equals(sCount,pCount)){
            results.add(0);
        }
        for (int i = 0; i < lenS-lenP; i++){
            --sCount[s.charAt(i)-'a'];
            ++sCount[s.charAt(i+lenP)-'a'];
            if (Arrays.equals(sCount,pCount)){
                results.add(i+1);
            }
        }
        return results;
    }

    /**
     * 350. 两个数组的交集 II（排序+双指针）
     */
    public int[] intersect_(int[] nums1, int[] nums2) {
        Arrays.sort(nums1);
        Arrays.sort(nums2);
        int index1 = 0, index2 = 0, index = 0;
        int len1 = nums1.length, len2 = nums2.length;
        int[] results = new int[len1+len2];
        while (index1 < len1 && index2 < len2){
            int num1 = nums1[index1];
            int num2 = nums2[index2];
            if (num1 == num2){
                results[index++] = num1;
                index1++;
                index2++;
            } else if (num1 < num2) {
                index1++;
            }else {
                index2++;
            }
        }
        return Arrays.copyOf(results,index);
    }

    /**
     * 350. 两个数组的交集 II（哈希）
     */
    public int[] intersect(int[] nums1, int[] nums2) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int num: nums1){
            map.put(num,map.getOrDefault(num,0)+1);
        }
        List<Integer> list = new ArrayList<>();
        for (int num:nums2){
            int count = map.getOrDefault(num,0);
            if (count > 0){
                list.add(num);
                map.put(num,map.get(num)-1);
            }
        }
        int len = list.size();
        int[] results = new int[len];
        for (int i = 0; i < len;i++){
            results[i] = list.get(i);
        }
        return results;
    }

    /**
     * 202. 快乐数
     */
    public boolean isHappy(int n) {
        if (n == 1) return true;
        Set<Integer> seen = new HashSet<>();
        while (true){
            int sum = 0;
            while (n > 0){
                int single = n % 10;
                n = n/10;
                sum += single*single;
            }
            n = sum;
            if (sum == 1) return true;
            if (seen.contains(sum)) return false;
            seen.add(sum);
        }
    }

    /**
     * 1. 两数之和（哈希）
     */
    public int[] twoSum_(int[] nums, int target) {
        Map<Integer,Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++){
            if (map.containsKey(target-nums[i])){
                return new int[]{map.get(target-nums[i]),i};
            }
            map.put(nums[i],i);
        }
        return null;
    }

    /**
     * 167. 两数之和 II - 输入有序数组
     */
    public int[] twoSum(int[] numbers, int target) {
        int len = numbers.length;
        for (int i = 0; i < len; i++){
            int remainder = target - numbers[i];
            int j = binarySearch_(numbers,remainder,i);
            if (j != -1){
                return new int[]{i+1, j+1};
            }
        }
        return null;
    }

    private int binarySearch_(int[] nums, int target,int i){
        int left = 0;
        int right = nums.length-1;
        while (left <= right){
            int middle = (left+right)/2;
            if (nums[middle] == target && middle != i){
                return middle;
            } else if (nums[middle] > target) {
                right = middle-1;
            }else {
                left = middle+1;
            }
        }
        return -1;
    }

    /**
     * 454. 四数相加 II
     */
    public int fourSumCount(int[] nums1, int[] nums2, int[] nums3, int[] nums4) {
        Map<Integer,Integer> map = new HashMap<>();
        for (int num1:nums1){
            for (int num2:nums2){
                map.put(num1+num2,map.getOrDefault(num1+num2,0)+1);
            }
        }
        int ans = 0;
        for (int num3:nums3){
            for (int num4:nums4){
                ans+=map.getOrDefault(-num3-num4,0);
            }
        }
        return ans;
    }

    /**
     * 15. 三数之和
     */
    public List<List<Integer>> threeSum_(int[] nums) {
        List<List<Integer>> results = new ArrayList<>();
        if (nums == null || nums.length < 3) return results;
        Arrays.sort(nums);
        int len = nums.length;
        for (int first = 0; first < len; ++first){
            if (first > 0 && nums[first] == nums[first-1]){
                continue;
            }
            int third = len-1;
            int target = -nums[first];
            for (int second = first+1; second < len; ++second){
                if (second > first+1 && nums[second] == nums[second-1]) continue;
                while (second < third && nums[second] + nums[third] > target){
                    --third;
                }
                if (second == third){
                    break;
                }
                if (nums[second]+nums[third] == target){
                    List<Integer> list = new ArrayList<>();
                    list.add(nums[first]);
                    list.add(nums[second]);
                    list.add(nums[third]);
                    results.add(list);
                }
            }
        }
        return results;
    }

    /**
     * 三数之和（暴力）
     */
    public List<List<Integer>> threeSum_brute(int[] nums){
        List<List<Integer>> results = new ArrayList<>();
        if (nums.length < 3){
            return results;
        }
        int len = nums.length;
        for (int first = 0; first < len; ++first){
            for (int second = first + 1; second < len; ++second){
                for (int third = second + 1; third < len; ++third){
                    if (nums[first] + nums[second] + nums[third] == 0){
                        List<Integer> list = new ArrayList<>();
                        list.add(nums[first]);
                        list.add(nums[second]);
                        list.add(nums[third]);
                        results.add(list);
                    }
                }
            }
        }
        //去重
        int n = results.size();
        List<List<Integer>> results_copy = new ArrayList<>();
        for (int i = 0; i < n; ++i){
            List<Integer> list = results.get(i);
            boolean flag = false;
            for (int j = 0; j < results_copy.size(); ++j){
                if (list.containsAll(results_copy.get(j)) && results_copy.get(j).containsAll(list)){
                    flag = true;
                    break;
                }
            }
            if (!flag){
                results_copy.add(list);
            }
        }
        return results_copy;
    }

    /**
     * 三数之和（暴力-重复优化）
     */
    public List<List<Integer>> threeSum_brute_repeat(int[] nums){
        List<List<Integer>> results = new ArrayList<>();
        int len = nums.length;
        if (len < 3){
            return results;
        }
        for (int first = 0; first < len; ++first){
            if (first > 0 && nums[first] == nums[first-1]){
                continue;
            }
            for (int second = first+1; second < len; ++second){
                if (second > first+1 && nums[second] == nums[second -1]){
                    continue;
                }
                for (int third = second + 1; third < len; ++ third){
                    if (third > second + 1 && nums[third] == nums[third-1]){
                        continue;
                    }
                    if (nums[first] + nums[second] + nums[third] == 0){
                        List<Integer> list = new ArrayList<>();
                        list.add(nums[first]);
                        list.add(nums[second]);
                        list.add(nums[third]);
                        results.add(list);
                    }
                }
            }
        }
        //去重
        int n = results.size();
        List<List<Integer>> results_copy = new ArrayList<>();
        for (int i = 0; i < n; ++i){
            List<Integer> list = results.get(i);
            boolean flag = false;
            for (int j = 0; j < results_copy.size(); ++j){
                if (list.containsAll(results_copy.get(j)) && results_copy.get(j).containsAll(list)){
                    flag = true;
                    break;
                }
            }
            if (!flag){
                results_copy.add(list);
            }
        }
        return results_copy;
    }

    /**
     * 15. 三数之和
     */
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> results = new ArrayList<>();
        if (nums.length < 3){
            return results;
        }
        int len = nums.length;
        Arrays.sort(nums);
        // 枚举a
        for (int first = 0; first < len; first++){
            //需要和上一次枚举的数不相同
            if (first > 0 && nums[first] == nums[first-1]){
                continue;
            }
            //c 对应的指针初始指向数组的最右端
            int third = len-1;
            int target = -nums[first];
            // 枚举b
            for (int second = first + 1; second < len; ++second){
                //需要和上一次枚举的数不相同
                if (second > first + 1 && nums[second] == nums[second - 1]){
                    continue;
                }
                // 需要保证b的指针在c的指针的左侧
                while (second < third && nums[second] + nums[third] > target){
                    --third;
                }
                // 如果指针重合，随着b后续的增加，就不会有满足a+b+c=0 并且 b<c 的 c了，可以退出循环
                if (second == third){
                    break;
                }
                if (nums[second] + nums[third] == target){
                    List<Integer> list = new ArrayList<>();
                    list.add(nums[first]);
                    list.add(nums[second]);
                    list.add(nums[third]);
                    results.add(list);
                }
            }
        }
        return results;
    }

    /**
     * 18. 四数之和
     */
    public List<List<Integer>> fourSum(int[] nums, int target) {
        List<List<Integer>> results = new ArrayList<>();
        if (nums == null || nums.length < 4) return results;
        Arrays.sort(nums);
        int len = nums.length;
        for (int first = 0; first < len; ++first){
            if (first > 0 && nums[first] == nums[first-1]){
                continue;
            }
            for (int second = first+1; second < len; ++second){
                if (second > first+1 && nums[second] == nums[second-1]){
                    continue;
                }
                int fourth = len-1;
                for (int third = second+1; third < len; ++third){
                    if (third > second+1 && nums[third] == nums[third-1]){
                        continue;
                    }
                    while (third < fourth && (long)nums[first]+nums[second]+nums[third]+nums[fourth] > target){
                        --fourth;
                    }
                    if (third == fourth){
                        break;
                    }
                    if ((long)nums[first]+nums[second]+nums[third]+nums[fourth] == target){
                        List<Integer> list = new ArrayList<>();
                        list.add(nums[first]);
                        list.add(nums[second]);
                        list.add(nums[third]);
                        list.add(nums[fourth]);
                        results.add(list);
                    }
                }
            }
        }
        return results;
    }

    /**
     * 344. 反转字符串
     */
    public void reverseString(char[] s) {
        int len = s.length;
        for (int i = 0; i < len/2; i++){
            char tmp = s[i];
            s[i] = s[len-i-1];
            s[len-i-1] = tmp;
        }
    }

    /**
     * 541. 反转字符串 II
     */
    public String reverseStr(String s, int k) {
        int n = s.length();
        char[] chars = s.toCharArray();
        for (int i = 0; i < n; i+=2*k){
            reverse(chars,i,Math.min(i+k,n)-1);
        }
        return new String(chars);
    }

    public void reverse(char[] chars,int left, int right){
        while (left < right){
            char tmp = chars[left];
            chars[left] = chars[right];
            chars[right] = tmp;
            left++;
            right--;
        }
    }


    /**
     * 剑指 Offer 05. 替换空格
     */
    public String replaceSpace__(String s) {
        return s.replace(" ", "%20");
    }

    /**
     * 剑指 Offer 05. 替换空格
     */
    public String replaceSpace_(String s) {
        StringBuilder sb = new StringBuilder();
        for (char c:s.toCharArray()){
            if (c == ' '){
                sb.append("%20");
            }else {
                sb.append(c);
            }
        }
        return sb.toString();
    }

    /**
     * 151. 反转字符串中的单词
     */
    public String reverseWords_(String s) {
        if (s == null || s.equals("")) return s;
        s = s.trim();
        Stack<String> stack = new Stack<>();
        String[] strings = s.split(" ");
        for (String str: strings){
            if (!str.equals("")) {
                stack.push(str);
            }
        }
        StringBuilder sb = new StringBuilder();
        while (!stack.isEmpty()){
            sb.append(stack.pop());
            sb.append(" ");
        }
        return sb.toString().trim();
    }

    /**
     * 151. 反转字符串中的单词
     */
    public String reverseWords(String s) {
        StringBuilder sb = removeSpace(s);
        StringBuilder stringReverse = reverseString(sb, 0, sb.length()-1);
        StringBuilder wordReverse = reverseEachWord(stringReverse);
        return wordReverse.toString();
    }

    private StringBuilder removeSpace(String s){
        int left = 0;
        int right = s.length()-1;
        //去除开头空格
        while (s.charAt(left) == ' '){
            left++;
        }
        //去除结尾空格
        while (s.charAt(right) == ' '){
            right--;
        }

        StringBuilder sb = new StringBuilder();
        while (left <= right){
            if (s.charAt(left) != ' ' || sb.charAt(sb.length()-1) != ' '){
                sb.append(s.charAt(left));
            }
            left++;
        }
        return sb;
    }

    private StringBuilder reverseEachWord(StringBuilder sb){
        int n = sb.length();
        int begin = 0;
        for (int i = 0; i < n; i++){
            if (sb.charAt(i) == ' ' || i == n-1){
                int end = i == n-1 ? i : i-1;
                while (begin < end){
                    char tmp = sb.charAt(begin);
                    sb.setCharAt(begin,sb.charAt(end));
                    sb.setCharAt(end,tmp);
                    begin++;end--;
                }
                begin = i+1;
            }
        }
        return sb;
    }
    private StringBuilder reverseString(StringBuilder sb, int start, int end){
        while (start < end){
            char tmp = sb.charAt(start);
            sb.setCharAt(start,sb.charAt(end));
            sb.setCharAt(end,tmp);
            start++;
            end--;
        }
        return sb;
    }

    /**
     * 剑指 Offer 58 - II. 左旋转字符串
     */
    public String reverseLeftWords(String s, int n) {
        int length = s.length();
        String subStr1 = s.substring(0,n);
        String subStr2 = s.substring(n,length);
        return subStr2+subStr1;
    }

    /**
     * 28. 找出字符串中第一个匹配项的下标
     */

    public int strStr__(String haystack, String needle) {
        int len1 = haystack.length();
        int len2 = needle.length();
        if (len2 > len1) return -1;
        for (int i = 0; i < len1-len2+1; i++){
            if (haystack.substring(i,i+len2).equals(needle)){
                return i;
            }
        }
        return -1;
    }

    /**
     * 28. 找出字符串中第一个匹配项的下标
     */

    public int strStr_(String haystack, String needle) {
        int len1 = haystack.length();
        int len2 = needle.length();
        if (len2 > len1) return -1;
        for (int i = 0; i < len1-len2+1; i++){
            int a = i,b = 0;
            while (b < len2 && haystack.charAt(a) == needle.charAt(b)){
                a++;
                b++;
            }
            if (b == len2) return i;
        }
        return -1;
    }

    /**
     * 28. 找出字符串中第一个匹配项的下标
     */

    public int strStr(String haystack, String needle) {
        int len1 = haystack.length();
        int len2 = needle.length();
        int[] next = new int[len2];
        for (int i = 1, j = 0; i < len2; i++){
            while (j > 0 && needle.charAt(i) != needle.charAt(j)){
                j = next[j-1];
            }
            if (needle.charAt(i) == needle.charAt(j)){
                j++;
            }
            next[i] = j;
        }
        for (int i = 0, j = 0; i < len1; i++){
            while (j > 0 && haystack.charAt(i) != needle.charAt(j)){
                j = next[j-1];
            }
            if (haystack.charAt(i) == needle.charAt(j)){
                j++;
            }
            if (j == len2){
                return i-len2+1;
            }
        }
        return -1;
    }

    /**
     * 459. 重复的子字符串
     */
    public boolean repeatedSubstringPattern__(String s) {
        int len = s.length();
        for (int i = 1; 2*i <= len; ++i){
            boolean match = true;
            if (len % i == 0){
                for (int j = i; j < len; ++j){
                    if (s.charAt(j) != s.charAt(j-i)){
                        match = false;
                        break;
                    }
                }
                if (match){
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * 459. 重复的子字符串
     */
    public boolean repeatedSubstringPattern_(String s) {
        return (s+s).indexOf(s,1) != s.length();
    }

    /**
     * 459. 重复的子字符串
     */
    public boolean repeatedSubstringPattern(String s) {
        String text = s+s;
        int n = text.length();
        int m = s.length();
        int[] next = new int[m];
        for (int i = 1, j = 0; i < m; ++i){
            while (j > 0 && s.charAt(j) != s.charAt(i)){
                j = next[j-1];
            }
            if (s.charAt(j) == s.charAt(i)){
                j++;
            }
            next[i] = j;
        }

        for (int i = 1, j = 0; i < n-1; ++i){
            while (j > 0 && text.charAt(i) != s.charAt(j)){
                j = next[j-1];
            }
            if (text.charAt(i) == s.charAt(j)){
                j++;
            }
            if (j == m){
                return true;
            }
        }
        return false;
    }

    /**
     * 509.斐波那契数
     */
    public int fib(int n) {
        int[] dp = new int[n+1];
        if (n == 0){
            return 0;
        }
        dp[0] = 0;
        dp[1] = 1;
        for (int i = 2; i <= n; i++){
            dp[i] = dp[i-1]+dp[i-2];
            System.out.println(dp[i]);
        }
        return dp[n];
    }

    /**
     * 70.爬楼梯
     */
    public int climbStairs(int n) {
        if (n < 2){
            return n;
        }
        int[] dp = new int[n+1];
        dp[1] = 1;
        dp[2] = 2;
        for (int i = 3; i <= n; i++){
            dp[i] = dp[i-1]+dp[i-2];
        }
        return dp[n];
    }

    /**
     * 746. 使用最小花费爬楼梯
     */
    public int minCostClimbingStairs(int[] cost) {
        int n = cost.length;
        int[] dp = new int[n+1];
        dp[0] = 0;
        dp[1] = 0;
        for (int i = 2; i <= n; i++){
            dp[i] = Math.min(dp[i-1]+cost[i-1], dp[i-2]+cost[i-2]);
        }
        return dp[n];
    }

    /**
     * 2460. 对数组执行操作
     */
    public int[] applyOperations(int[] nums) {
        int n = nums.length;
        for (int i = 0,j = 0; i < n; i++){
            //翻倍
            if (i+1 < n && nums[i] == nums[i+1]){
                nums[i] = nums[i]*2;
                nums[i+1] = 0;
            }
            //交换
            if (nums[i] != 0){
                int tmp = nums[i];
                nums[i] = nums[j];
                nums[j] = tmp;
                j++;
            }
        }
        return nums;
    }

    private void printArray(int[] nums){
        for (int i = 0; i < nums.length; i++){
            System.out.print(nums[i] + " ");
        }
    }


    /**
     * 选择排序
     * @param nums 待排序数组
     * @return
     */
    public int[] selectSort(int[] nums){
        int len = nums.length;
        for (int i = 0; i < len; i++){
            int index = i;
            for (int j = i+1; j < len; j++){
                if (nums[j] < nums[index]){
                    index = j;
                }
            }
            int tmp = nums[i];
            nums[i] = nums[index];
            nums[index] = tmp;
        }
        return nums;
    }

    /**
     * 62. 不同路径（动态规划）
     */
    public int uniquePaths_(int m, int n) {
        int[][] f = new int[m][n];
        //边界处理
        for (int i = 0; i < m; i++){
            f[i][0] = 1;
        }
        for (int j = 0; j < n; j++){
            f[0][j] = 1;
        }

        for (int i = 1; i < m; i++){
            for (int j = 1; j < n; j++){
                f[i][j] = f[i-1][j] + f[i][j-1];
            }
        }
        return f[m-1][n-1];
    }

    /**
     * 62. 不同路径（排列组合）
     */
    public int uniquePaths(int m, int n) {
        //从大到小算会出现精度问题
//        double ans = 1;
//        for (int y = m-1; y > 0; y--){
//            ans = ans * (n+y-1) / y;
//        }
//        return (int) ans;
        long ans = 1;
        for (int y = 0; y < m-1;++y) {
            ans = ans * (n + y) / (y+1);
        }
        return (int) ans;
    }

    /**
     * 63. 不同路径 II
     */
    public int uniquePathsWithObstacles_(int[][] obstacleGrid) {
        int m = obstacleGrid.length;
        int n = obstacleGrid[0].length;
        if (obstacleGrid[m-1][n-1] == 1 || obstacleGrid[0][0] == 1){
            return 0;
        }
        int[][] df = new int[m][n];
        for (int i = 0; i < m && obstacleGrid[i][0] == 0; i++){
            df[i][0] = 1;
        }
        for (int j = 0; j < n && obstacleGrid[0][j] == 0; j++){
            df[0][j] = 1;
        }
        for (int i = 1; i < m; i++){
            for (int j = 1; j < n; j++){
                if (obstacleGrid[i][j] == 0){
                    df[i][j] = df[i-1][j]+df[i][j-1];
                }
            }
        }
        return df[m-1][n-1];
    }

    /**
     * 63.不同路径II
     * @param obstacleGrid
     * @return
     */
    public int uniquePathsWithObstacles(int[][] obstacleGrid){
        int m = obstacleGrid.length;
        int n = obstacleGrid[0].length;
        int[][] dp = new int[m][n];

        //初始化dp，第一列都为1，只能向下走，障碍及障碍以下路径为0
        for (int i = 0; i < m && obstacleGrid[i][0] == 0; i++){
            dp[i][0] = 1;
        }
        //第一列都为1，只能向右走，障碍及障碍右边路径为0
        for (int j = 0; j < n && obstacleGrid[0][j] == 0; j++){
            dp[0][j] = 1;
        }
        for (int i = 1; i < m; i++){
            for (int j = 1; j < n; j++){
                if (obstacleGrid[i][j] == 1){//有障碍，则路径为0
                    dp[i][j] = 0;
                }else{
                    dp[i][j] = dp[i-1][j]+dp[i][j-1];
                }
            }
        }
        return dp[m-1][n-1];
    }

    /**
     * 343. 整数拆分
     */
    public int integerBreak_(int n) {
        int[] dp = new int[n+1];
        for (int i = 2; i <= n; i++){
            for (int j = 1; j < i; j++){
                dp[i] = Math.max(dp[i], Math.max(j*(i-j), j*dp[i-j]));
            }
        }
        return dp[n];
    }

    /**
     * 343. 整数拆分
     * @param n
     * @return
     */
    public int integerBreak(int n){
        int[] dp = new int[n+1];
        for (int i = 2; i < n; i++){
            for (int j = 1; j < i; j++){
                dp[i] = Math.max(dp[i-1], Math.max(j*(i-j), j*dp[i-j]));
            }
        }
        return dp[n];
    }

    /**
     * 96. 不同的二叉搜索树（动态规划）
     */
    public int numTrees_(int n) {
        int[] dp = new int[n+1];
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i <= n; i++){
            for (int j = 1; j <= i; j++){
                dp[i] += dp[j-1]*dp[i-j];
            }
        }
        return dp[n];
    }

    /**
     * 96. 不同的二叉搜索树（卡塔兰数）
     */
    public int numTrees(int n) {
        long c = 1;
        for (int i = 0; i < n; i++){
            c = c * 2 * (2 * i + 1)/( i + 2);
        }
        return (int) c;
    }

    /**
     * 94. 二叉树的中序遍历
     */
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> list = new ArrayList<>();
        inorder(root,list);
        return list;
    }

    private void inorder(TreeNode root, List<Integer> list){
        if (root == null){
            return;
        }
        inorder(root.left, list);
        list.add(root.val);
        inorder(root.right, list);
    }

    /**
     * 494. 目标和(回溯法)
     */
    int count = 0;
    public int findTargetSumWays_(int[] nums, int target) {
        backtrack(nums,target,0,0);
        return count;
    }

    private void backtrack(int[] nums, int target, int index, int sum){
        if (index == nums.length){
            if (sum == target){
                count++;
            }
        }else {
            backtrack(nums, target, index+1, sum+nums[index]);
            backtrack(nums, target, index+1, sum-nums[index]);
        }
    }

    /**
     * 494. 目标和(动态规划)
     */
    public int findTargetSumWays(int[] nums, int target) {
        int sum = 0;
        for (int num : nums){
            sum+=num;
        }
        //判断nums之和是否符合规则
        int diff = sum - target;
        if (diff < 0 || diff % 2 != 0){
            return 0;
        }

        int n = nums.length, neg = diff/2;

        int[][] dp = new int[n+1][neg+1];

        dp[0][0] = 1;
        for (int i = 1; i <= n; i++){
            int num = nums[i];
            for (int j = 0; j <= neg; j++){
                dp[i][j] = dp[i-1][j];
                if (j >= num){
                    dp[i][j] += dp[i-1][j-num];
                }
            }
        }
        return dp[n][neg];
    }

    /**
     * 474. 一和零
     */
    public int findMaxForm(String[] strs, int m, int n) {
        int length = strs.length;
        int[][][] dp = new int[length+1][m+1][n+1];
        for (int i = 1; i <= length; i++){
            int[] zerosOnes = getZerosOnes(strs[i-1]);
            for (int j = 0; j <= m; j++){
                for (int k = 0; k <= n; k++){
                    dp[i][j][k] = dp[i-1][j][k];
                    if (j >= zerosOnes[0] && k >= zerosOnes[1]){
                        dp[i][j][k] = Math.max(dp[i-1][j][k], dp[i-1][j-zerosOnes[0]][k-zerosOnes[1]]+1);
                    }
                }
            }
        }
        return dp[length][m][n];
    }

    public int[] getZerosOnes(String str){
        int[] zerosOnes = new int[2];
        int length = str.length();
        for (int i = 0; i < length; i++){
            zerosOnes[str.charAt(i) - '0']++;
        }
        return zerosOnes;
    }

    /**
     * 14. 最长公共前缀
     */
    public String longestCommonPrefix(String[] strs) {
        int n = strs.length;
        int minLen = Integer.MAX_VALUE;
        for (String s : strs) {
            if (s.length() < minLen) {
                minLen = s.length();
            }
        }
        boolean flag = true;//是否存在公共前缀
        String maxPrefix = "";
        for (int i = 0; i <= minLen; i++){
            String str = strs[0].substring(0,i);
            for (int j = 1; j < n; j++){
                if (!strs[j].startsWith(str)){
                    flag = false;
                    break;
                }
            }

            if (flag){
                maxPrefix = str;
            }
        }
        return maxPrefix;
    }


    public static int win1(int[] arr){
        if (arr == null || arr.length == 0){
            return 0;
        }
        int first = f(arr,0,arr.length-1);
        int second = g(arr,0,arr.length-1);
        return Math.max(first,second);
    }

    public static int f(int[] arr, int L, int R){
        if (L == R){
            return arr[L];
        }
        int p1 = arr[L] + g(arr, L+1, R);
        int p2 = arr[R] + g(arr, L, R-1);
        return Math.max(p1,p2);
    }

    public static int g(int[] arr, int L, int R){
        if (L == R){
            return 0;
        }

        int p1 = f(arr, L+1, R);//对手拿走了L位置的数
        int p2 = f(arr, L, R-1);//对手拿走了R位置的数
        return Math.min(p1,p2);
    }

    public static int win2(int[] arr){
        if (arr == null || arr.length == 0){
            return 0;
        }

        int N = arr.length;
        int[][] fmap = new int[N][N];
        int[][] gmap = new int[N][N];
        for (int i = 0; i < N; i++){
            for (int j = 0; j < N; j++){
                fmap[i][j] = -1;
                gmap[i][j] = -1;
            }
        }
        int first = f2(arr, 0, arr.length-1, fmap, gmap);
        int second = g2(arr, 0, arr.length-1, fmap, gmap);
        return Math.max(first,second);
    }

    public static int f2(int[] arr, int L, int R, int[][] fmap, int[][] gmap){
        if (fmap[L][R] != -1){
            return fmap[L][R];
        }
        int ans = 0;
        if (L == R){
            ans = arr[L];
        }else {
            int p1 = arr[L] + g2(arr, L+1, R, fmap, gmap);
            int p2 = arr[R] + g2(arr, L, R-1, fmap, gmap);
            ans = Math.max(p1,p2);
        }
        fmap[L][R] = ans;
        return ans;
    }

    public static int g2(int[] arr, int L, int R, int[][] fmap, int[][] gmap){
        if (gmap[L][R] != -1){
            return gmap[L][R];
        }

        int ans = 0;
        if (L != R){
            int p1 = f2(arr, L+1, R, fmap, gmap);
            int p2 = f2(arr, L, R-1, fmap, gmap);
            ans = Math.min(p1,p2);
        }
        gmap[L][R] = ans;
        return ans;
    }

    public static int win3(int[] arr){
        if (arr == null || arr.length == 0){
            return 0;
        }

        int N = arr.length;
        int[][] fmap = new int[N][N];
        int[][] gmap = new int[N][N];
        for (int i = 0; i < N; i++){
            fmap[i][i] = arr[i];
        }

        int startRow = 0;
        for (int startCol = 1; startCol < N; startCol++){
            int row = 0;
            int col = startCol;
            while (col < N){
                fmap[row][col] = Math.max(arr[row]+gmap[row+1][col], arr[col]+gmap[row][col-1]);
                gmap[row][col] = Math.min(fmap[row+1][col], fmap[row][col-1]);
                row++;
                col++;
            }
        }

        return Math.max(fmap[0][N-1],gmap[0][N-1]);
    }

    //所有的货，重量和价值，都在w和v数组里
    //为了方便，其中没有负数
    //bag背包容量，不能超过这个载重
    //返回：不超重的情况下，能够得到的最大价值
    public static int maxValue(int[] w, int[] v, int bag){
        return process(w,v,0,bag);
    }

    //当前考虑到了index号货物，index...所有的货物可以自由选择

    //做的选择不能超过背包容量
    //返回最大价值
    public static int process(int[] w, int[] v, int index, int bag){
        if (bag < 0){
            return -1;
        }
        if (index == w.length){
            return 0;
        }
        //有货，index位置的货
        //bag有空间，0
        //不要当前的货
        int p1 = process(w,v,index+1,bag);
        int p2 = 0;
        int next = process(w,v,index+1,bag-w[index]);
        if (next != -1){
            p2 = v[index]+next;
        }
        return Math.max(p1,p2);
    }

    public static int dp(int[] w, int[] v, int bag){
        int N = w.length;
        int[][] dp = new int[N+1][bag+1];

        for (int index = N-1; index >= 0; index--){
            for (int rest = 0; rest <= bag; rest++){
                int p1 = dp[index+1][rest];//不要当前货物
                int p2 = 0;
                int next = rest - w[index] < 0 ? -1 : dp[index+1][rest-w[index]];
                if (next != -1){
                    p2 = v[index]+next;
                }
                dp[index][rest] = Math.max(p1,p2);
            }
        }
        return dp[0][bag];
    }

    static class Myqueue{

        Stack<Integer> stackin = new Stack<>();
        Stack<Integer> stackout = new Stack<>();
        void push(int x){
            stackin.push(x);
        }

        int pop(){
            if (stackout.empty()){
                while (!stackin.empty()){
                    stackout.push(stackin.pop());
                }
            }
            return stackout.pop();
        }

        int peek(){
            if (stackout.empty()){
                while (!stackin.empty()){
                    stackout.push(stackin.pop());
                }
            }
            return stackout.pop();
        }

        boolean empty(){
            return stackout.empty() && stackin.empty();
        }
    }

    /**
     * 21.合并两个有序链表
     */
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        if(list1 == null){
            return list2;
        }
        if (list2 == null){
            return list1;
        }
        ListNode prehead = new ListNode(-1);
        ListNode pre = prehead;
        while (list1 != null && list2 != null){
            if (list1.val <= list2.val){
                pre.next = list1;
                list1 = list1.next;
            }else {
                pre.next = list2;
                list2 = list2.next;
            }
            pre = pre.next;
        }
        pre.next = list1 == null ? list2:list1;
        return prehead.next;
    }

    /**
     * 面试题 02.07. 链表相交
     */
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if(headA == null || headB == null){
            return null;
        }
        ListNode pA = headA;
        ListNode pB = headB;
        while (pA != pB){
            pA = pA == null? headB:pA.next;
            pB = pB == null? headA:pB.next;
        }
        return pA;
    }











}
