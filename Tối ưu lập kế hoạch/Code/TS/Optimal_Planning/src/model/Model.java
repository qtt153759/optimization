package model;

import java.io.*;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.*;


public class Model {
	final int TIME_LIMIT_IN_MINUTE = 30;
	boolean isTimeUp;
	ScheduledExecutorService timer;
	
	
	List<List<Integer>> bestPartitions;
	double weighted;
	double timeElapse;
	String status;
	int violation;
	int alpha;
	int k;
	
	private static int NUM_GENERATION;
	private int numVer;
    private int tbl=3;
    Solution currSol, bestSol, lastImprovedSol;
    private final int W=1000;
    
    Data data_input;
    	
	String modelName;
	
	private static final String inputFolder ="data/input/";
    private static final String outputFolder="data/output/";
	
	public Model(){
        this.modelName = "TabuSearch";
    }
	public void readInput(String filename) throws IOException{
		data_input = new Data(filename);
	}
	public void readInput(File file) throws IOException{
		data_input = new Data(file);
	}
	
	public String getModelName() {
        return modelName;
    }
	void run(String dataName, int k, int alpha){
        System.out.printf("Start Solving %s with k=%d, alpha=%d\n",dataName,k,alpha);

        timer = Executors.newSingleThreadScheduledExecutor();
        timer.schedule(new Timer(this), TIME_LIMIT_IN_MINUTE, TimeUnit.MINUTES);
        isTimeUp=false;

        this.timeElapse = System.currentTimeMillis();
        this.solve(k,alpha);
        this.timeElapse = System.currentTimeMillis()- this.timeElapse;

        insightResult(k,alpha);
        stop();
    }
	void insightResult(int k, int alpha){
        this.violation=0;
        if (status.equals("INFEASIBLE")){
            this.violation= Utils.calViolationOfWholePartitions(this.bestPartitions,k,alpha);
        }
    }
	void stop(){
        timer.shutdownNow();
        isTimeUp = true;
    }
	
	private Solution genSolution(){
        final int CAP_PART = (int)Math.floor((double) numVer/(double) k)+ alpha;
        List<List<Integer>> partitions = new ArrayList<>(k);
        for (int i=0;i<k;++i){
            partitions.add(new ArrayList<>(CAP_PART));
        }

        Random  rd = new Random();
        for (int i=0;i<numVer;++i){
            partitions.get(rd.nextInt(k)).add(i);
        }

        return new Solution(partitions);
    }
	
	static void setNumGeneration(int numVer){
        final int factor = numVer/36;
        NUM_GENERATION = (factor+1)*100;
    }
	
	void solve(int k, int alpha) {
        Solution.weightedMatrix = data_input.getWeightedMatrix();

        numVer = data_input.getNumVertices();
        this.k=Solution.k=k;
        this.alpha = Solution.alpha= alpha;
        Solution.W = W;

        int[] tabu = new int[numVer];
        final int TB_MIN=2, TB_MAX=5;

        currSol = genSolution();
        double bestObj = Double.POSITIVE_INFINITY;
        double oldObj;

        setNumGeneration(numVer);
        int it=0;
        int stable=0, stableLimit=30;
        int restartFreq=100;

        while ((it<NUM_GENERATION)&&(!isTimeUp)){
            it++;
            if (currSol.obj<bestObj){
                bestObj = currSol.obj;
                bestSol = (Solution)currSol.clone();
                stable = 0;
            } else if (stable == stableLimit){
                currSol = (Solution) lastImprovedSol.clone();
                stable=0;
            } else {
                stable++;
                if (it%restartFreq==0){
                    currSol = genSolution();
                    Arrays.fill(tabu,0);
                }
            }

            oldObj=currSol.obj;
            Move moveToNext = currSol.findBestNeighbor(tabu);
            if (moveToNext.desPartition==-1||
                moveToNext.srcPartition==-1||
                moveToNext.idxVertex==-1||
                moveToNext.posVertex==-1){
                currSol = genSolution();
                continue;
            }
            currSol.update(moveToNext);

            for (int i = 0; i< tabu.length; ++i){
                if (tabu[i]>0){
                    tabu[i]--;
                }
            }

            tabu[moveToNext.idxVertex]= tbl;
            if (currSol.obj < oldObj){
                if (tbl>TB_MIN){
                    tbl--;
                }

                lastImprovedSol = (Solution) currSol.clone();
                stable=0;
            } else {
                if (tbl<TB_MAX){
                    tbl++;
                }
            }
        }

        // return result
        this.bestPartitions = bestSol.partitions;
        this.status = this.bestSol.getStatus();
        this.weighted = bestSol.getCutWeight();
    }
	void setupResultFile(String filename,boolean isAppend){
        char bracket = isAppend?']':'[';
        try {
            System.out.println("Setting up file "+filename);
            PrintWriter printWriter = new PrintWriter(new OutputStreamWriter(new FileOutputStream(filename,isAppend)));
            printWriter.printf("%c",bracket);
            printWriter.close();
        } catch (FileNotFoundException fileNotFoundException){
            System.out.println(fileNotFoundException.getMessage());
            fileNotFoundException.printStackTrace();
        }
    }
	void writeResult(String filename, int k, int alpha, int runIdx, char delimiter){
        try {
            PrintWriter printWriter = new PrintWriter(new OutputStreamWriter(new FileOutputStream(filename,true)));
            printWriter.printf("%c\n{\n", delimiter);
            printWriter.printf("%2s\"K\": %d,\n", "", k);
            printWriter.printf("%2s\"Alpha\": %d,\n", "", alpha);
            printWriter.printf("%2s\"runIdx\": %d,\n", "", runIdx);
            printWriter.printf("%2s\"weight\": %f,\n", "", this.weighted);
            printWriter.printf("%2s\"violation\": %d,\n", "", this.violation);
            printWriter.printf("%2s\"timeElapsed\": %f,\n", "", this.timeElapse);
            printWriter.printf("%2s\"partitions\": {", "");

            int i=0; String deli="";
            for (List<Integer> parti : this.bestPartitions){
                printWriter.printf("%s\n%4s\"%d\": %s", deli, "", i++, parti.toString());
                deli=",";
            }

            printWriter.printf("\n%2s}\n",""); // close bracket for partition field
            printWriter.printf("}"); // cloe bracket for this object
            printWriter.close(); 
            bestPartitions.clear(); // clear to store result of new run(if nRun > 1)
        } catch (FileNotFoundException fileNotFoundException){
            System.out.println(fileNotFoundException.getMessage());
            fileNotFoundException.printStackTrace();
        }
    }
	public static void execute(Model model, int nRun,int[] kArray, int[]alphaArray, String... dataTypes){
        StringBuilder inputDir = new StringBuilder(inputFolder);
        StringBuilder outputFileName = new StringBuilder(outputFolder + model.getModelName() + "/");
        for (String dataType: dataTypes){
            inputDir.append(dataType).append("/");
            outputFileName.append(dataType).append("/");
        }


        File dir = new File(inputDir.toString());



        for (File f : Objects.requireNonNull(dir.listFiles())) {
//		File[] files = {new File("data/input/test/data_12_distance.txt")};
//		for (File f: files){
            String dataName;
            if (!f.isDirectory() && (dataName = f.getName()).endsWith(".txt")) {
                dataName = dataName.substring(0, dataName.lastIndexOf(".txt")); // remove postfix .txt
                outputFileName.append(dataName).append(".json");

                try {
                    model.readInput(f);
                    model.setupResultFile(outputFileName.toString(),false);
                    char delimiter=' ';
                    for (int k : kArray){
                        for (int alpha: alphaArray) {
                            for (int i = 0; i < nRun; ++i) {
                                model.run(dataName,k,alpha);

                                model.writeResult(outputFileName.toString(),k,alpha,i,delimiter); delimiter=',';
                            }
                        }
                    }
                    model.setupResultFile(outputFileName.toString(),true);
                } catch (IOException ioException){
                    System.out.println("Error when reading data from "+dataName);
                    System.out.println(ioException.getMessage());
                    ioException.printStackTrace();
                }

                // remove the name of the previous result file , 5 is the length of extension ".json"
                outputFileName.delete(outputFileName.length()-dataName.length()-5,outputFileName.length());
            }
        }
    }
	
}

class Timer implements Runnable{
    Model model;

    public Timer(Model model){
        this.model=model;
    }

    public void run() {
        model.stop();
    }
}
