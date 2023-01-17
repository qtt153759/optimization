package model;

public class Move {
	public int srcPartition;
	public int desPartition;
	public int posVertex;
	public int idxVertex;
	public double obj;
	
	public Move(int srcPartition, int desPartition, int posVertex, int idxVertex, double obj) {
		this.desPartition = desPartition;
		this.idxVertex = idxVertex;
		this.obj = obj;
		this.posVertex = posVertex;
		this.srcPartition = srcPartition;
	}
}
