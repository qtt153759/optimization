package model;

public class Config {
    public static int[][] kArray = new int[][] {{2,3,4},{4,5,6},{6,7,8},{2,3,5,7}};
    public static int[][] alphaArray = new int[][] {{1,2},{5,6},{10,11},{4,5}};

    public static Pair<int[],int[]> getParameter(String dataSize){
        return switch (dataSize) {
            case "small" -> new Pair<>(kArray[0],alphaArray[0]);
            case "medium" -> new Pair<>(kArray[1],alphaArray[1]);
            case "huge" -> new Pair<>(kArray[2],alphaArray[2]);
            default -> new Pair<>(kArray[3],alphaArray[3]);
        };
    }
} 
class Pair<T, S>{
	T num1;
	S num2;
	public Pair(T num1, S num2) {
		this.num1 = num1;
		this.num2 = num2;
	}
}