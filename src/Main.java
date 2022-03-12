import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class Main {

    public static void main(String[] args) {

        Main main = new Main();
//        int[] security = {1,2,5,4,1,0,2,4,5,3,1,2,4,3,2,4,8};
//        int[] security = {5,3,3,3,5,6,2};
//        int time = 2;
//        System.out.println(main.goodDaysToRobBank_presum(security,time));
        String s = "AAAAAAAAAAA";
        System.out.println(main.findRepeatedDnaSequences(s));

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
    public List<Integer> goodDaysToRobBank_presum(int[] security,int time){
        List<Integer> results = new ArrayList<>();
        int n = security.length;
        int[] non = new int[n];
        for(int i = 1; i < n;i++){
            if(security[i-1] == security[i]) continue;
            non[i] = security[i-1] < security[i] ? 1 : -1;//1代表非递减，-1代表非递增
        }
        int[] nonincre = new int[n+1];//非递增个数
        int[] nondecre = new int[n+1];//非递减个数
        for (int i = 1;i <= n;i++){
            nonincre[i] = nonincre[i-1]+(non[i-1] == 1 ? 1:0);
            nondecre[i] = nondecre[i-1]+(non[i-1] == -1?1:0);
        }
        for (int i = time;i < n-time;i++){
            int c1 = nonincre[i+1]-nonincre[i+1-time];
            int c2 = nondecre[i+1+time]-nondecre[i+1];
            if(c1 == 0 && c2 == 0){
                results.add(i);
            }
        }
        return results;
    }


    /**
     * 187. 重复的DNA序列
     * @param s DNA序列
     * @return 重复子序列
     */
    public List<String> findRepeatedDnaSequences(String s) {
        int num = 10;
        List<String> results = new ArrayList<>();
        char[] chars = s.toCharArray();
        int n = s.length();
        HashMap<String,Integer> map = new HashMap<>();
        for (int i = 0; i <=n-num;i++){
            int count = map.getOrDefault(s.substring(i,i+num),0);
            if(count == 1) results.add(s.substring(i,i+10));
            map.put(s.substring(i,i+num),++count);
        }
        return results;
    }


     //Definition for singly-linked list.
     public class ListNode {
         int val;
         ListNode next;
         ListNode() {}
         ListNode(int val) { this.val = val; }
         ListNode(int val, ListNode next) { this.val = val; this.next = next; }
     }
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode sum = null;
        if(sum != null){
            sum.next = new ListNode(addVal(l1).val+addVal(l2).val);
        }else{
            sum = new ListNode(addVal(l1).val+addVal(l2).val);
        }
        return sum;
    }

    public ListNode addVal(ListNode node){
        if(node.next == null){
            return node;
        }else {
            return addVal(node.next);
        }
    }
}
