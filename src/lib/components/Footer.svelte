<script>
	import MainNav from './MainNav.svelte'
	import { siteAuthor } from '$lib/config'
</script>

<footer>
	<p>Created with <a href="https://github.com/josh-collinsworth/sveltekit-blog-starter">sveltekit-blog-starter.</a></p>
	<p>&copy;{new Date().getFullYear()} {siteAuthor}</p>
</footer>