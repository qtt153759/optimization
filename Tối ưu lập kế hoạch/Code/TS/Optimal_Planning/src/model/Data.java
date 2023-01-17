package model;

import java.util.List;
import java.util.stream.Collectors;
import java.io.*;
import java.util.*;

public class Data {
	private final int NUM_VERTICES;
	private final int NUM_EDGES;
	private List<Edge> listEdges;
	
	public Data(String fileName) throws IOException {
        this(new File(fileName));
    }
	
	public Data(File file) throws IOException {
        BufferedReader bfReader = new BufferedReader(new InputStreamReader(new FileInputStream(file)));
        StringTokenizer stringTokenizer = new StringTokenizer(bfReader.readLine()," ");

        NUM_VERTICES = Integer.parseInt(stringTokenizer.nextToken());
        NUM_EDGES = Integer.parseInt(stringTokenizer.nextToken());
        listEdges = new ArrayList<>(NUM_EDGES);

        String currLine;
        while ((currLine= bfReader.readLine())!=null){
            stringTokenizer = new StringTokenizer(currLine);
            listEdges.add(new Edge(
                    Integer.parseInt(stringTokenizer.nextToken()),
                    Integer.parseInt(stringTokenizer.nextToken()),
                    Double.parseDouble(stringTokenizer.nextToken()))
            );
        }
        bfReader.close();
    }
	
	public int getNumVertices() {
        return NUM_VERTICES;
    }
	public int getNumEdges() {
        return NUM_EDGES;
    }
	public List<List<Double>> getWeightedMatrix() {
        List<List<Double>> weightedMatrix = new ArrayList<>(NUM_VERTICES);
        double[][] matrix = new double[NUM_VERTICES][NUM_VERTICES];
        int firstVer, secVer;
        for (Edge row: listEdges) {
            firstVer = row.vertex1;
            secVer = row.vertex2;
            matrix[firstVer][secVer]=matrix[secVer][firstVer] = row.cost;
        }
        for (double[] row : matrix){
            weightedMatrix.add(Arrays.stream(row).boxed().collect(Collectors.toList()));
        }
        return weightedMatrix;
    }
}

class Edge{
	public int vertex1;
	public int vertex2;
	public double cost;
	
	public Edge(int v1, int v2, double cost) {
		this.cost = cost;
		this.vertex1 = v1;
		this.vertex2 = v2;
	}
}