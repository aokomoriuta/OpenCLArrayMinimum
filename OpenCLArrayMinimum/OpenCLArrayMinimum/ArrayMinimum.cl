/* 最小値を探索する */
/*!
	\param elementCount 入力配列の要素数
	\param values 配列
*/
__kernel void ArrayMinimum(
	const uint elementCount,
	__global float* values)
{
	/* グローバルIDを取得 */
	size_t gi = get_global_id(0);

	/* 入力要素数より大きいワークアイテム番号なら何もしない */
	if(gi > elementCount)
	{
		return;
	}

	/* 割る数を2倍ずつ、要素数まで繰り返す */
	for(int n = 2; n < 2*elementCount; n *= 2)
	{
		/* 計算する要素番号で、隣が存在すれば */
		if((gi%n == 0) && (gi + n/2 < elementCount)) 
		{
			/* 隣の要素と小さい方設定する */
			values[gi] = min(values[gi], values[gi + n/2]);
		}

		/* ここまで同期 */
		barrier(CLK_GLOBAL_MEM_FENCE);
	}

}