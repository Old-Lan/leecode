public class Course {
    public static void main(String[] args) {
//        Integer i = new Integer("23");
//        Integer i = new Integer(23);
//        Integer i = Integer.valueOf("23");
//        Integer i = Integer.parseInt("23", 8);
//        Double d = new Double();0
//        Double d = Double.valueOf("23.45");
//        int i = (Integer.valueOf("23").intValue());
//        double d = (Double.valueOf("23.4")).doubleValue();
//        int i = (Double.valueOf("23.4")).intValue();
//        String s = (Double.valueOf("23.4")).toString();
        String s1 = "Welcome to Java";

        char[] chars = s1.toCharArray();
        System.out.println(chars);
    }
    private String courseName;
    private String[] students = new String[100];
    private int numberOfStudents;

    public Course(String courseName){
        this.courseName = courseName;
    }

    public void addStudent(String student){
        students[numberOfStudents] = student;
        numberOfStudents++;
    }

    public String[] getStudents() {
        return students;
    }

    public int getNumberOfStudents() {
        return numberOfStudents;
    }

    public String getCourseName() {
        return courseName;
    }
}
