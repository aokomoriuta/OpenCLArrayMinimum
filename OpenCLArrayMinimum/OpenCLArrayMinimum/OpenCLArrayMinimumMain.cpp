﻿#include "OpenCLArrayMinimumMain.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <CL/cl.hpp>
#include <boost/random.hpp>
#include <boost/timer.hpp>

#include "Exception.hpp"

using std::cout;
using std::endl;
using std::vector;
using std::string;

namespace ArrayMinimum
{
	//! アプリケーションを実行する
	int ArrayMinimumMain::Main()
	{
		try
		{
			// 計算を実行
			ArrayMinimumMain::Compute();
		}
		catch(cl::Error error)
		{
			// エラーを作成
			LWisteria::CL::Exception::Exception exception = LWisteria::CL::Exception::Exception(error);

			cout << "== エラー！ == " << endl
				 << ";場所" << endl
				 << ": " << exception.Where() << endl
				 << ";詳細" << endl
				 << ": " << exception.Message() << endl;
		}

		// 成功で終了
		system("pause");
		return 0;
	}

	//! 計算する
	void ArrayMinimumMain::Compute()
	{
		// 実行開始
		cout << "= 最小値テスト =" << endl;

		// OpenCL Cソースファイル名
		const string filepath = "ArrayMinimum.cl";

		// エントリポイント名
		const string entryPoint = "ArrayMinimum";

		// ベクトルの要素数
		const cl_uint elementCount = 7000;

		// 最大値と最小値
		const cl_float minValue = -0;
		const cl_float maxValue = +10000;

/*****************/
		// 要素数を表示
		cout << ";要素数" << endl
			 << ": " << elementCount << endl;
		

/*****************/
		cout << endl
			 << "# 入力配列の初期化" << endl;

		// 入力値を初期化
		cl_float* input = new cl_float[elementCount]; 

		// 乱数生成器の作成
		boost::minstd_rand gen(42);
		boost::uniform_real<> dst(minValue, maxValue);
		boost::variate_generator<
			boost::minstd_rand&, boost::uniform_real<>
		> random(gen, dst);
		random();

		// 全要素について
		for(cl_uint i = 0; i < elementCount; i++)
		{
			// 入力値をランダムで与える
			input[i] = (cl_float)random();
		}

		// OpenCL用出力
		cl_float outputCL;

		// CPU用出力
		cl_float outputCPU;

		// タイマーで時間測定
		boost::timer timer = boost::timer();


/*****************/
		cout << "# CPUの計算";
		timer.restart();

		// 最小値を入力値の最初の値で設定
		outputCPU = input[0];
		int t = 0;
		// 全要素について
		for(cl_uint i = 1; i < elementCount; i++)
		{
			// 小さい方を最小値にする
			outputCPU = min(outputCPU, input[i]);

			if(outputCPU == input[i])
			{
				t = i;
			}
		}
		cout << " - " << timer.elapsed() << "[s]" << endl;
	
/*****************/
		cout << "# プラットフォームを取得" << endl;

		// プラットフォーム一覧を取得
		vector<cl::Platform> platforms = vector<cl::Platform>();
		cl::Platform::get(&platforms);

		// 先頭のプラットフォームを取得
		cl::Platform platform = platforms[0];


/*****************/
		cout << "# デバイス一覧を取得" << endl;
		
		// デバイス一覧を取得
		vector<cl::Device> devices = vector<cl::Device>();
		platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

		// 先頭のデバイスを取得
		cl::Device device = devices[0];


/*****************/
		cout << "# コンテキストを作成" << endl;

		// コンテキストを作成
		cl::Context context = cl::Context(devices);


/*****************/
		cout << "# コマンドキューを作成" << endl;
		
		// コマンドキューを作成
		cl::CommandQueue queue = cl::CommandQueue(context, device);


/*****************/
		cout << "# バッファーの作成" << endl;

		// バッファーを読み書き可能にして作成
		cl::Buffer bufferValues = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * elementCount);


/*****************/
		cout << "# ソースの読み込み" << endl;

		// 入力ストリームの作成
		std::ifstream sourceFileStream = std::ifstream(filepath);
		
		// ソースファイルの作成
		std::stringstream sourceStringStream = std::stringstream();

		// ファイルの終端まで
		while(!sourceFileStream.eof())
		{
			// 一行読み込み
			char buffer[1024];
			sourceFileStream.getline(buffer, 1024);

			// ソースファイルに追加
			sourceStringStream << buffer;
		}

		// 先頭の不要な文字を除去
		std::string sourceString = sourceStringStream.str();
		sourceString = sourceString.substr(3);

		// ソースの作成
		cl::Program::Sources sources = cl::Program::Sources();
		sources.push_back(std::make_pair(sourceString.c_str(), sourceString.size()));


/*****************/
		cout << "# プログラムの作成" << endl;

		// プログラムを作成
		cl::Program program = cl::Program(context, sources);


/*****************/
		cout << "# プログラムをビルド" << endl;

		try
		{
			// コンテキスト内の全てのデバイスを対象にしてプログラムをビルド
			program.build(devices);
		}
		// OpenCL例外があった場合
		catch(cl::Error error)
		{
			// ビルドエラーなら
			if(error.err() == CL_BUILD_PROGRAM_FAILURE)
			{
				// ビルドログを表示
				cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
			}

			// エラーをそのまま投げる
			throw error;
		}

/*****************/
		cout << "# カーネルを作成" << endl;

		// カーネルを作成
		cl::Kernel kernel = cl::Kernel(program, entryPoint.c_str());


/*****************/
		cout << "# 引数を設定" << endl;

		// 引数を設定
		// # 入力配列の要素数
		// # 入力配列
		kernel.setArg(0, elementCount);
		kernel.setArg(1, bufferValues);

/*****************/
		cout << endl
			 << "== 計算の実行 ==" << endl
		     << "# デバイスにデータを書き込み" << endl;
		timer.restart();

		// 非同期で入力値を書き込み
		queue.enqueueWriteBuffer(bufferValues, CL_FALSE, 0, sizeof(cl_float) * elementCount, input);

/*****************/
		cout << "# カーネルの実行" << endl;

		// カーネルを実行
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(elementCount), cl::NullRange);

/*****************/
		cout << "# デバイスからデータを読み込み" << endl;

		// 同期で出力値に読み込み
		queue.enqueueReadBuffer(bufferValues, CL_TRUE, 0, sizeof(cl_float), &outputCL);

		cout << "#: " << timer.elapsed() << "[s]" << endl;

		cl_float* debug = new cl_float[elementCount];
		queue.enqueueReadBuffer(bufferValues, CL_TRUE, 0, sizeof(cl_float)*elementCount, debug);
		cout << debug[t];

/*****************/
		cout << endl
			 << "== 結果 == " << endl
			 << "# CLでの計算  = " << outputCL << endl
			 << "# CPUでの計算 = " << outputCPU << endl;


/*****************/
		delete[] input;
	}
}
#undef foreach