import { sveltekit } from '@sveltejs/kit/vite'

/** @type {import('vite').UserConfig} */
const config = {
	base: "https://boxxfish.github.io",
	plugins: [sveltekit()],
	server: {
		fs: {
			allow: ['.']
		}
	}
};

export default config
