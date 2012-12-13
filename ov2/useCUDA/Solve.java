import com.sun.jna.Library;
import com.sun.jna.Native;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class Solve {

    /* 静的変数：このクラス内のすべてのオブジェクトで共有される(実体は1つ )*/
    public static int N = 72;//Number of　object
    public static double Lx = 26.0;//area
    public static double Ly = 15.0;//area
    public static int interval = 0;
    public static double time ;
    public static double dt = 0.05;//Time step
    public static double a = 3.0; //Sensitivity
    public static double b = 4.0;
    public static double c = 0.0; //-1<= c <=1
    public static double d = 1.0;
    public static double h = 0.5;
    public static double flow = 0.0; 
    public static double range = 2.0; 
    public static int typeCode = 2; //interaction type
    public static int initCode  = 1; 
    
    public static double[] x; // position_x  0 <= x < LX
    public static double[] y; // position_y  0 <= y < LY
    public static double[] vx; // velocity_x
    public static double[] vy;// velocity_y 
    public static double[] v;  
    public static double[][] dx; // -0.5*LX<= dx <= 0.5*LX
    public static double[][] dy; // -0.5*LY<= dy <= 0.5*LY 
    public static double[][] dr;  
    public static int[][] pointer;
    double[] kx1, kx2, kx3, kx4;
    double[] ky1, ky2, ky3, ky4;
    double[] kvx1, kvx2, kvx3, kvx4;
    double[] kvy1, kvy2, kvy3, kvy4;
    double[] sx, sy, svx, svy;
    
    File file;
    FileWriter filewriter;
    public static int counter;

    /* use C library */
    public interface TRF_API extends Library {
	
	TRF_API INSTANCE = (TRF_API)
	    Native.loadLibrary("libtraffic_sync.so", TRF_API.class);  
	
	void trf_init2Dim(int N);
	void trf_OV2Dim(double x[], double y[], double vx[], double vy[], double Lx, double Ly, double dt, double a, double b, double c, double d, double h, double flow, double range, int typeCode);
	void trf_exit2Dim();
	
    }




    Solve(){
	init();
    }

    // ---------------------------- init -----------------------------------------------
    void init(){
	time = 0.0;
	kx1=new double[N];kx2=new double[N];kx3=new double[N];kx4=new double[N];
	ky1=new double[N];ky2=new double[N];ky3=new double[N];ky4=new double[N];
	kvx1=new double[N];kvx2=new double[N];kvx3=new double[N];kvx4=new double[N];
	kvy1=new double[N];kvy2=new double[N];kvy3=new double[N];kvy4=new double[N];
	sx=new double[N];sy=new double[N];svx=new double[N];svy=new double[N];
	x = new double[N]; 
	y = new double[N]; 
	vx = new double[N];
	vy = new double[N];
	v = new double[N];
	dx = new double[N][N];
	dy = new double[N][N];
	dr = new double[N][N];
	pointer = new int[N][6]; //point 6-neighbors of each object

	try{
	    file = new File("/home/hirabayashi/research/test/OV_sim/2dim/useCUDA/sync/data/data.dat");
	    if(checkBeforeWritefile(file))
		filewriter = new FileWriter(file);
	}catch(IOException e){
	    System.out.println(e);
	}


	// set initial potision & velocity
	if(initCode == 1){
	    //uniform start
	    double low = Math.sqrt((double)N/2);
	    double mx = Lx/low;
	    double my = (Ly-1)/low;
	    x[0]= mx * 0.5;
	    y[0]= my * 0.5;
	    x[N/2]= mx;
	    y[N/2]= my;
	    
	    for(int i=1;i<N/2;i++){
		x[i]=x[i-1]+mx;
		y[i]=y[i-1];
		if(x[i]>Lx){
		    x[i] = mx * 0.5;
		    y[i] += my;
		}
		x[N/2+i]=x[N/2+i-1]+mx;
		y[N/2+i]=y[N/2+i-1];
		if(x[N/2+i]>Lx){
		    x[N/2+i] = mx;
		    y[N/2+i] += my;
		}
	    }
	    for(int i=0;i<N;i++){
		vx[i] = 1.0;
		vy[i] = 0.0;
	    }
	}else if(initCode ==2){
	    //random start
	    for(int i=0;i<N;i++){
		x[i] = Lx * Math.random();
		y[i] = Ly * Math.random();
		vx[i] = 1.0-2.0*Math.random();
		vy[i] = 1.0-2.0*Math.random();
	    }
	}	



    }
    // -------------------------- init -------------------------------------------

    //	 dr calculation
    void calc_dr(double x[],double y[]){
	for(int i=0;i<N;i++){
	    for(int j=N-1;j>0;j--){
		dx[i][j] = x[j]-x[i];
		dy[i][j] = y[j]-y[i];
		//periodic boudary condition
		if(dx[i][j] > Lx*0.5) {
		    dx[i][j] = dx[i][j] -Lx;
		}else if(dx[i][j] <= -Lx*0.5){
		    dx[i][j] = Lx + dx[i][j];	
		}
		if(dy[i][j] >Ly*0.5) {
		    dy[i][j] = dy[i][j] -Ly ;
		}else if(dy[i][j] < -Ly*0.5){
		    dy[i][j] = Ly + dy[i][j];	
		} 
		dr[i][j] = Math.sqrt(dx[i][j]*dx[i][j]+dy[i][j]*dy[i][j]);
		dx[j][i] = -dx[i][j];
		dy[j][i] = -dy[i][j];
		dr[j][i] = dr[i][j];
	    }
	}
    }


    void calc_v(){
	for(int i=0;i<N;i++){
	    v[i]=Math.sqrt(vx[i]*vx[i]+vy[i]*vy[i]);
	}
    }


    void func(double x[],double y[],double vx[], double vy[],double fx[],double fy[],double fvx[],double fvy[]){	
	double sumForceX = 0.0;
	double sumForceY = 0.0;
	calc_dr(x,y);
	calc_v();
	// OV equation
	if(typeCode ==1){
	    for(int i=0;i<N;i++){
		//	boolean sdcode = true;
		for(int j=0;j<N;j++){
		    if(i==j) continue;
		    if(dr[i][j] > range) continue;
		    sumForceX += OV(dr[i][j]) *(1.0+cos(i,j)) *0.5* nx(i,j);
		    sumForceY += OV(dr[i][j]) *(1.0+cos(i,j)) *0.5* ny(i,j);
		    /*進行方向に±πに粒子があるか確認
		      if(sdcode == true){
		      double theta = Math.atan2(dy[i][j],dx[i][j]) - Math.atan2(vy[i],vx[i]); //-Math.PI< theta<=Math.PI
		      if(theta > Math.PI){
		      theta = theta - 2.0*Math.PI;
		      }else if(theta <= -Math.PI){
		      theta = theta + 2.0*Math.PI;
		      }
		      if(-Math.PI*0.5 <= theta&&theta <= Math.PI*0.5) sdcode = false;
		      }
		    */
		}
		//if(sdcode==true){
		sumForceX +=0.75*vx[i]/v[i]; //self-driving force
		sumForceY +=0.75*vy[i]/v[i]; //self-driving force	
		//}
		fvx[i] = a * ( sumForceX-vx[i] + flow);
		fvy[i] = a * ( sumForceY-vy[i]);
		sumForceX = 0.0;
		sumForceY = 0.0;
	    }
	}else if(typeCode ==2){
	    //	type-2 : interact with 6-neighbors
	    calc_neighbors();
	    for(int i=0;i<N;i++){
		for(int j=0;j<6;j++){
		    if(i==j) continue;
		    if(pointer[i][j]==-1) continue;
		    sumForceX += OV(dr[i][pointer[i][j]]) * (1.0+cos(i,pointer[i][j]))*0.5 * nx(i,pointer[i][j]);
		    sumForceY += OV(dr[i][pointer[i][j]]) * (1.0+cos(i,pointer[i][j]))*0.5 * ny(i,pointer[i][j]);
		}
		//	if(pointer[i][0]==-1&&pointer[i][1]==-1&&pointer[i][2]==-1){	//進行方向±π/2に粒子がない場合
		sumForceX +=0.75*vx[i]/v[i]; //self-driving force
		sumForceY+=0.75*vy[i]/v[i]; //self-driving force	
		//	}
		fvx[i] = a * ( sumForceX-vx[i] + flow);
		fvy[i] = a * ( sumForceY-vy[i]);	
		sumForceX = 0.0;
		sumForceY = 0.0;
	    }
	}
	for(int i=0;i<N;i++){
	    fx[i] = vx[i];
	    fy[i] = vy[i];
	}
    }


    void calc_neighbors(){
	double theta;
	for(int i=0;i<N;i++){
	    for(int j=0;j<6;j++){
		pointer[i][j]= -1;
	    }
	}
	for(int i=0;i<N;i++){
	    for(int j=0;j<N;j++){
		if(i==j)continue;
		if(dr[i][j] > range) continue;
		theta = Math.atan2(dy[i][j],dx[i][j]) - Math.atan2(vy[i],vx[i]); //-Math.PI< theta<=Math.PI
		if(theta > Math.PI){
		    theta = theta - 2.0*Math.PI;
		}else if(theta <= -Math.PI){
		    theta = theta + 2.0*Math.PI;
		}
		if(-Math.PI/2.0 <=theta&&theta< -Math.PI/6.0){					//1
		    if(pointer[i][0]==-1) pointer[i][0]=j;
		    else if(dr[i][j] < dr[i][pointer[i][0]]) pointer[i][0]=j;
		}else if(-Math.PI/6.0 <=theta&&theta< Math.PI/6.0){			//2 :進行方向±π/6
		    if(pointer[i][1]==-1) pointer[i][1]=j;						
		    else if(dr[i][j] < dr[i][pointer[i][1]]) pointer[i][1]=j;
		}else if(Math.PI/6.0 <=theta&&theta< Math.PI/2.0){			//3
		    if(pointer[i][2]==-1) pointer[i][2]=j;
		    else if(dr[i][j] < dr[i][pointer[i][2]]) pointer[i][2]=j;
		}else if(Math.PI/2.0 <=theta&&theta< 5.0*Math.PI/6.0){		//4
		    if(pointer[i][3]==-1) pointer[i][3]=j;
		    else if(dr[i][j] < dr[i][pointer[i][3]]) pointer[i][3]=j;
		}else if(5.0*Math.PI/6.0 <=theta||theta< -5.0*Math.PI/6.0){	//5
		    if(pointer[i][4]==-1) pointer[i][4]=j;
		    else if(dr[i][j] < dr[i][pointer[i][4]]) pointer[i][4]=j;
		}else if( -5.0*Math.PI/6.0 <=theta&&theta< -Math.PI/2.0){		//6
		    if(pointer[i][5]==-1) pointer[i][5]=j;
		    else if(dr[i][j] < dr[i][pointer[i][5]]) pointer[i][5]=j;
		}
	    }
	}
    }
    
    
    //OV function
    double OV(double dr){
	if(dr<=range) return h * (tanh (b*(dr-d))+c);
	else return 0.0;
    }

    double tanh(double x){
	return (Math.exp(2.0*x)-1.0)/(Math.exp(2.0*x)+1.0);
    }

    double cos(int i,int j){
	if(i==j) return 0.0;
	else return (dx[i][j]*vx[i]+dy[i][j]*vy[i])/(dr[i][j]*v[i]);
    }

    double nx(int i, int j){
	if(i==j) return 0.0;
	else return dx[i][j]/dr[i][j];
    }

    double ny(int i, int j){
	if(i==j) return 0.0;
	return dy[i][j]/dr[i][j];
    }
    
    
    //Runge_kutta solution
    void rungekutta(double x[],double y[],double vx[],double vy[]){

	//System.out.println("typeCode_Solve = " + typeCode);
	TRF_API.INSTANCE.trf_OV2Dim(x, y, vx, vy, Lx, Ly, dt, a, b, c, d, h, flow, range, typeCode);

	try{
	    if(checkBeforeWritefile(file)){
		if(counter<100){//100計算分、車[0]の座標と速度を出力
		    //filewriter.write(x[0] + " " + y[0] + " " + vx[0] + " " + vy[0] + '\n');
		    filewriter.write(String.format("%.8f %.8f %.8f %.8f\n", x[0], y[0], vx[0], vy[0]));
		    counter++;
		}else if(counter == 100){
		    System.out.println("data output completed!!\n");
		    filewriter.write(String.format("N = %d typeCode = %d\n", N, typeCode));	
		    filewriter.write("counter = 100\n");
		    counter++;
		}else{
		    filewriter.close();  // 100回出力したら、ストリームをクローズ
		}
	    }
	}catch(IOException e){
	    System.out.println(e);
	}

	/*	func(x, y, vx, vy, kx1, ky1, kvx1, kvy1);
		for(int i=0;i<N;i++){
		sx[i] = x[i] +  kx1[i]*dt*0.5;
		sy[i] = y[i] +  kx1[i]*dt*0.5;
		svx[i] = vx[i] +  kvx1[i]*dt*0.5;
		svy[i] = vy[i] +  kvy1[i]*dt*0.5;
		}
		func(sx, sy, svx, svy, kx2, ky2, kvx2, kvy2);
		for(int i=0;i<N;i++){
		sx[i] = x[i] +  kx2[i]*dt*0.5;
		sy[i] = y[i] +  kx2[i]*dt*0.5;
		svx[i] = vx[i] +  kvx2[i]*dt*0.5;
		svy[i] = vy[i] +  kvy2[i]*dt*0.5;
		}
		func(sx, sy, svx, svy, kx3, ky3, kvx3, kvy3);
		for(int i=0;i<N;i++){
		sx[i] = x[i] +  kx3[i]*dt;
		sy[i] = y[i] +  ky3[i]*dt;
		svx[i] = vx[i] +  kvx3[i]*dt;
		svy[i] = vy[i] +  kvy3[i]*dt;
		}
		func(sx, sy, svx, svy, kx4, ky4, kvx4, kvy4);
		for(int i=0;i<N;i++){
		x[i] += (kx1[i]+2.0*kx2[i]+2.0*kx3[i]+kx4[i])/6.0*dt;
		y[i] += (ky1[i]+2.0*ky2[i]+2.0*ky3[i]+ky4[i])/6.0*dt;
		vx[i] += (kvx1[i]+2.0*kvx2[i]+2.0*kvx3[i]+kvx4[i])/6.0*dt;
		vy[i] += (kvy1[i]+2.0*kvy2[i]+2.0*kvy3[i]+kvy4[i])/6.0*dt;
		//periodic boudary condition
		if(x[i]<0.0){
		x[i] = x[i] + Lx;
		}else if (x[i]>=Lx){
		x[i] = x[i] - Lx;
		}
		if(y[i]<0.0){
		y[i] = y[i] + Ly;
		}else if(y[i]>=Ly){
		y[i] = y[i] - Ly;
		}
		}	*/
    }
    
    private static boolean checkBeforeWritefile(File file){
	if(file.exists()){	
	    if(file.isFile() && file.canWrite()){
		return true;
	    }
	}
	return false;
    }
    

}
