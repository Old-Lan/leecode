import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

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
        int[] nums = {1,0,1,0,1};
        System.out.println(main.longestOnes_(nums,2));

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

    public List<Integer> goodDaysToRobBank_(int[] security,int time){
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
}
