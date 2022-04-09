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
        int[] w = {3,14,1,7};
        Main main1 = new Main(w);
        while (true){
            System.out.println( main1.pickIndex_());
        }

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
}
