package zb;

public class Father {
    public static void main(String[] args) {
        Father f = new Father();
        f.sayHello();
    }
    public void sayHello(){
        System.out.println("Hello");
    }

    public void sayHello(String name){
        System.out.println("Hello" + " " + name);
    }
}
