public class Test1{
    private String text;

    public Test1(String s){
        String text = s;
    }

    public static void main(String[] args) {
        //10.10.1
//        String s1 = "Welcome to Java";
//        String s2 = s1;
//        String s3 = new String("Welcome to Java");
//        String s4 = "Welcome to Java";
//
//        System.out.println(s1 == s2);
//        System.out.println(s1 == s3);
//        System.out.println(s1 == s4);
//        System.out.println(s1.equals(s3));
//        System.out.println(s1.equals(s4));
//        System.out.println("Welcome to Java".replace("Java", "HTML"));
//        System.out.println(s1.replace('o', 'T'));
//        System.out.println(s1.replaceAll("o", "T"));
//        System.out.println(s1.replaceFirst("o", "T"));
//        System.out.println(s1.toCharArray());
        //10.10.8
//        Test test = new Test("ABC");
//        System.out.println(test.text.toLowerCase());
        //10.10.10
//        System.out.println("Hi, ABC, good".matches("ABC"));
//        System.out.println("Hi, ABC, good".matches("ABC"));
//        System.out.println("A,B;C".replaceAll(",;","#"));
//        System.out.println("A,B;C".replaceAll("[,;]", "#"));
//
//        String[] tokens = "A,B;C".split("[,;]");
//        for (int i = 0; i < tokens.length; i++){
//            System.out.println(tokens[i] + " ");
//        }
        String s = "Hi, Good Morning";
        System.out.println(m(s));

    }

    public static int m(String s){
        int count = 0;
        for (int i = 0; i < s.length(); i++)
            if (Character.isUpperCase(s.charAt(i)))
                count++;
        return count;
    }
}