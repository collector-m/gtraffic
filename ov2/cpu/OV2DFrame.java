import java.awt.*;
import java.awt.event.*;
import javax.swing.*;

public class OV2DFrame extends JFrame implements ActionListener,Runnable{
	private static final long serialVersionUID = 1L;
	static final int WIDTH = 1040;
	static final int HEIGHT = 750;
	Solve so = new Solve();
	Thread th = null;
	
	public OV2DFrame(){
		//Window setting
		setTitle("2-dimensional OV Model");
		setSize(WIDTH,HEIGHT);
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		Container c = getContentPane();
		c.add(new Paint());
		/*--GUI Components Setting--*/
		//ComboBox
		JComboBox N_cb = new JComboBox();
		JComboBox a_cb = new JComboBox();
		JComboBox dt_cb = new JComboBox();
		JComboBox c_cb = new JComboBox();
		JComboBox flow_cb = new JComboBox();
		JComboBox typeCode_cb = new JComboBox();
		JComboBox initCode_cb = new JComboBox();
		N_cb.addItem(Integer.toString(Solve.N));
		for(int i=2;i<=25;i++){
			String s = String.valueOf(2*i*i);
			N_cb.addItem(s);
		}
		a_cb.addItem(Double.toString(Solve.a));
		for(double i=0.1;i<=5.0;i+=0.1){
			String s = String.valueOf((float)i);
			a_cb.addItem(s);
		}
		c_cb.addItem(Double.toString(Solve.c));
		for(double i=1.0;i>=0.1;i-=0.1){
			String s = String.valueOf((float)i);
			c_cb.addItem(s);
		}
		for(double i=0;i>=-1.0;i-=0.1){
			String s = String.valueOf((float)i);
			c_cb.addItem(s);
		}
		dt_cb.addItem(Double.toString(Solve.dt));
		dt_cb.addItem("0.05"); dt_cb.addItem("0.1"); dt_cb.addItem("0.01");dt_cb.addItem("0.005");dt_cb.addItem("0.001");
		flow_cb.addItem(Double.toString(Solve.flow));
		for(double i=2.0;i>=0.2;i-=0.2){
			String s = String.valueOf((float)i);
			flow_cb.addItem(s);
		}
		for(double i=0.0;i>=-2.0;i-=0.2){
			String s = String.valueOf((float)i);
			flow_cb.addItem(s);
		}
		if(Solve.typeCode==1){
			typeCode_cb.addItem("all");typeCode_cb.addItem("6-neighbors");
		}else if(Solve.typeCode==2){
			typeCode_cb.addItem("6-neighbors");typeCode_cb.addItem("all");
		}
		if(Solve.initCode==1){
			initCode_cb.addItem("uniform");initCode_cb.addItem("random");
		}else if(Solve.initCode==2){
			initCode_cb.addItem("random");initCode_cb.addItem("uniform");
		}	
		N_cb.setActionCommand("N");
		a_cb.setActionCommand("a");  
		c_cb.setActionCommand("c");
		dt_cb.setActionCommand("dt"); 
		flow_cb.setActionCommand("flow");
		typeCode_cb.setActionCommand("type"); 
		initCode_cb.setActionCommand("init"); 
		N_cb.addActionListener(this);
		a_cb.addActionListener(this);
		dt_cb.addActionListener(this);
		c_cb.addActionListener(this);
		flow_cb.addActionListener(this);
		typeCode_cb.addActionListener(this);
		initCode_cb.addActionListener(this);
		//button
		JButton run_bt = new JButton("Run/Stop");
		JButton reset_bt = new JButton("Reset");
		JButton onestep_bt = new JButton("1-Step");
		run_bt.setActionCommand("run");  
		reset_bt.setActionCommand("reset");
		onestep_bt.setActionCommand("onestep");
		run_bt.addActionListener(this);
		reset_bt.addActionListener(this);
		onestep_bt.addActionListener(this);
		//LayoutManagement
		JPanel p = new JPanel();
		JPanel p1 = new JPanel();
		JPanel p2 = new JPanel();
		p1.add(new JLabel("N =")); p1.add(N_cb);
		p1.add(new JLabel("a =")); p1.add(a_cb);
		p1.add(new JLabel("c =")); p1.add(c_cb);
		p1.add(new JLabel("dt =")); p1.add(dt_cb);
		p1.add(new JLabel("flow =")); p1.add(flow_cb);
		p2.add(new JLabel("interaction")); p2.add(typeCode_cb);
		p2.add(new JLabel("init value")); p2.add(initCode_cb);
		p2.add(run_bt);
		p2.add(reset_bt);
		p2.add(onestep_bt);
		p.setLayout(new GridLayout(2,1));
		p.add(p1);
		p.add(p2);
		c.add(p,BorderLayout.NORTH);	
	}
	void init(){
		so.init();
	}
	public void run(){
		while(th != null){
			so.rungekutta( Solve.x, Solve.y, Solve.vx, Solve.vy);
			Solve.time += Solve.dt;
			repaint();
		//	output();
			try{Thread.sleep(Solve.interval);}
			catch(InterruptedException e){}
		}
	}
	void output(){
		System.out.println("x "+Solve.x[0]+"y "+Solve.y[0]);
		System.out.println("vx "+Solve.vx[0]+"vy "+Solve.vy[0]);
		for(int i=0;i<6;i++){
			System.out.println(Solve.pointer[0][i]);
		}
	}
	public void actionPerformed(ActionEvent e){
		if(e.getActionCommand() == "N"){
			th = null;
			JComboBox source_N = (JComboBox)e.getSource();
			String item_N = (String) source_N.getSelectedItem();
			Solve.N = Integer.parseInt(item_N);
			init();	
		}else if(e.getActionCommand() == "a"){
			JComboBox source_a = (JComboBox)e.getSource();
			String item_a = (String) source_a.getSelectedItem();
			Solve.a = Double.parseDouble(item_a);
		}else if(e.getActionCommand() == "c"){
			JComboBox source_c = (JComboBox)e.getSource();
			String item_c = (String) source_c.getSelectedItem();
			Solve.c = Double.parseDouble(item_c);
		}else if(e.getActionCommand() == "dt"){
			JComboBox source_dt = (JComboBox)e.getSource();
			String item_dt = (String) source_dt.getSelectedItem();
			Solve.dt = Double.parseDouble(item_dt);
		}else if(e.getActionCommand() == "flow"){
			JComboBox source_flow = (JComboBox)e.getSource();
			String item_flow = (String) source_flow.getSelectedItem();
			Solve.flow = Double.parseDouble(item_flow);
		}else if(e.getActionCommand() == "type"){
			JComboBox source_type = (JComboBox)e.getSource();
			String item_type = (String) source_type.getSelectedItem();
			if(item_type=="all"){
				Solve.typeCode = 1;
			}else if(item_type=="6-neighbors"){
				Solve.typeCode = 2;
			}
		}else if(e.getActionCommand() == "init"){
			JComboBox source_init = (JComboBox)e.getSource();
			String item_init = (String) source_init.getSelectedItem();
			if(item_init=="uniform"){
				Solve.initCode = 1;
			}else if(item_init=="random"){
				Solve.initCode = 2;
			}
			init();
		}else if(e.getActionCommand() == "run"){
			if(th == null){
				th = new Thread(this);
				th.start();
			}else{
				th = null;
			}
		}else if(e.getActionCommand() == "reset"){
			th = null;
			init();
		}else if(e.getActionCommand() == "onestep"){
			th = null;
			Solve.time += Solve.dt;
			so.rungekutta( Solve.x, Solve.y, Solve.vx, Solve.vy);
		}
		repaint();	
	}
}
