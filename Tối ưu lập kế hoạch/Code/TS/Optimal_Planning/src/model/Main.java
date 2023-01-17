package model;

import model.Solution;
import model.Model;

public class Main {
	public static void main(String[] args) {
		Args args_model = parseArgument(args);
		Pair<int[], int[]> modelParams = Config.getParameter(args[2]);
		
		Model.execute(args_model.model, args_model.nRun, modelParams.num1, modelParams.num2, args[2]);
		System.out.println("a");
	}
	private static Args parseArgument(String[] args){
        if (args.length < 3) {
            System.out.println("Not Enough Parameter");
            return null;
        } 

        int nRun;
        try {
            nRun = Integer.parseInt(args[1]);
        } catch (NumberFormatException numberFormatException){
            nRun=1;
        }

        return new Args(new Model(), nRun, args[2]);
    }
}
class Args{
	Model model;
	int nRun;
	String data_size;
	
	public Args(Model model, int nRun, String data_size) {
		this.data_size = data_size;
		this.model = model;
		this.nRun = nRun;
	}
}
